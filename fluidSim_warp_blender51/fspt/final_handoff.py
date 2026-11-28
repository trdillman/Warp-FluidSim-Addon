"""Identity-bound main-thread retirement before allocating an evaluated Final.

The render worker may wait for this explicit Final lifecycle operation. It must
not hold the viewport admission lock while waiting. Optional producers observe
the device hold until the caller releases its completed Final claim. This
module owns requests only; Blender and GPU teardown belong to the callback.
"""
from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from threading import Event, Lock, current_thread, main_thread


class FinalHandoffError(RuntimeError):
    pass


class FinalHandoffTimeout(FinalHandoffError):
    pass


@dataclass(frozen=True)
class FinalRetirementReceipt:
    device: str
    request_id: object
    reserved_bytes: int
    generation_count: int
    owners_retired: bool


@dataclass(eq=False)
class _Request:
    device: str
    retire: object
    identity: object = field(default_factory=object)
    done: Event = field(default_factory=Event)
    state: str = 'PENDING'
    receipt: FinalRetirementReceipt | None = None
    error: str = ''
    cancelled: bool = False
    released: bool = False


class _FinalClaim:
    def __init__(self, queue, request):
        self._queue, self._request = queue, request
        self.receipt = request.receipt

    def release(self):
        self._queue._release(self._request)


class _FailureHold:
    """Suppression only: this token never authorizes a Final allocation."""

    receipt = None

    def __init__(self, queue, device, identity):
        self._queue, self._device, self._identity = queue, device, identity

    def release(self):
        self._queue._release_failure(self._device, self._identity)


class FinalHandoff:
    """One Final owner per device, with a main-thread-only cleanup service."""

    def __init__(self):
        self._lock = Lock()
        self._requests = deque()
        self._holders = {}
        self._failure_holds = {}
        self._counts = dict(requests=0, serviced=0, timeouts=0, failures=0)

    def blocked(self, device):
        with self._lock:
            return str(device) in self._holders or str(device) in self._failure_holds

    def retain_failure(self, device):
        """Retain an uncertain owner independently of queued or active Finals.

        The caller owns the failed backend strongly and releases this token
        only after checked lifecycle retirement. Multiple failures cannot
        replace each other or a successful/running request's device claim.
        """
        device, identity = str(device), object()
        with self._lock:
            self._failure_holds.setdefault(device, set()).add(identity)
        return _FailureHold(self, device, identity)

    def _release_failure(self, device, identity):
        with self._lock:
            identities = self._failure_holds.get(device)
            if identities is not None:
                identities.discard(identity)
                if not identities:
                    del self._failure_holds[device]

    @property
    def pending_count(self):
        with self._lock:
            return sum(request.state == 'PENDING' for request in self._requests)

    @property
    def statistics(self):
        with self._lock:
            return dict(self._counts)

    def _release(self, request):
        with self._lock:
            request.released = True
            # A timed-out cleanup may still be executing on main. Its device
            # remains held until that exact callback returns, even though the
            # requesting worker has already refused to allocate Final geometry.
            if request.state != 'RUNNING' and self._holders.get(request.device) is request:
                del self._holders[request.device]
            if request.state != 'RUNNING':
                try:
                    self._requests.remove(request)
                except ValueError:
                    pass
                request.retire = None

    def _execute(self, request):
        if current_thread() is not main_thread():
            raise RuntimeError('Final retirement service requires the main thread')
        with self._lock:
            if (request.state != 'PENDING' or request.cancelled
                    or self._holders.get(request.device) is not request):
                return False
            request.state = 'RUNNING'
            self._counts['serviced'] += 1
        receipt, error = None, ''
        try:
            receipt = request.retire(request.device, request.identity)
            if (not isinstance(receipt, FinalRetirementReceipt)
                    or receipt.device != request.device or receipt.request_id is not request.identity
                    or type(receipt.reserved_bytes) is not int or receipt.reserved_bytes != 0
                    or type(receipt.generation_count) is not int or receipt.generation_count != 0
                    or receipt.owners_retired is not True):
                raise FinalHandoffError('Final retirement receipt is stale, wrong-device or nonempty')
        except BaseException as exc:
            # The resource owner retains failed teardown resources. Do not
            # preserve a callback traceback as an accidental lifetime policy.
            error = f'{type(exc).__name__}: {exc}'
        with self._lock:
            if error:
                request.error = error
                self._counts['failures'] += 1
            request.receipt = receipt if not error and not request.cancelled else None
            request.retire = None
            request.state = 'CANCELLED' if request.cancelled else ('FAILED' if error else 'READY')
            if request.released and self._holders.get(request.device) is request:
                del self._holders[request.device]
            request.done.set()
        return True

    def acquire(self, device, retire, *, timeout=15.0):
        """Return a claim only after matching zero-ownership proof.

        Enter the viewport coordinator's Final boundary first. Release the
        returned claim after leaving that boundary and closing the Final backend.
        """
        if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
            raise ValueError('Final handoff timeout must be finite and positive')
        if not callable(retire):
            raise TypeError('Final handoff requires an owned retirement callback')
        device = str(device)
        request = _Request(device, retire)
        on_main = current_thread() is main_thread()
        with self._lock:
            if device in self._holders or device in self._failure_holds:
                raise FinalHandoffError('Another Final already owns this device handoff')
            self._holders[device] = request
            self._counts['requests'] += 1
            if not on_main:
                self._requests.append(request)
        try:
            if on_main:
                self._execute(request)
            elif not request.done.wait(timeout):
                with self._lock:
                    request.cancelled = True
                    if request.state != 'RUNNING':
                        request.state = 'CANCELLED'
                    self._counts['timeouts'] += 1
                raise FinalHandoffTimeout('Main-thread Final retirement did not complete before its deadline')
            with self._lock:
                if request.state != 'READY' or request.cancelled or request.receipt is None:
                    raise FinalHandoffError(request.error or 'Final retirement request was cancelled')
                if self._holders.get(device) is not request:
                    raise FinalHandoffError('Final retirement request lost its device ownership')
                if device in self._failure_holds:
                    raise FinalHandoffError('A failed owner appeared during Final retirement')
            return _FinalClaim(self, request)
        except BaseException:
            self._release(request)
            raise

    def service(self):
        """Process queued requests at the top of the persistent main timer."""
        if current_thread() is not main_thread():
            raise RuntimeError('Final retirement service requires the main thread')
        count = 0
        while True:
            with self._lock:
                if not self._requests:
                    return count
                request = self._requests.popleft()
            count += int(self._execute(request))

    def cancel_pending(self, reason):
        """Cancel waiting work at lifecycle teardown; active Finals stay held."""
        with self._lock:
            for request in tuple(self._holders.values()):
                if request.state not in {'PENDING', 'RUNNING'}:
                    continue
                request.cancelled = True
                request.error = str(reason)
                if request.state == 'PENDING':
                    request.state = 'CANCELLED'
                request.done.set()
            self._requests = deque(r for r in self._requests if r.state == 'PENDING')


MAIN_FINAL_HANDOFF = FinalHandoff()
