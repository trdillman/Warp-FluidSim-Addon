# Deck navigation and click routing

Base: `665bd48aaf02522db5e69f014ea0b66f79b3a886` (`origin/master`, freshly fetched
3 November 2026). Original canonical checkout and its unrelated dirty work were
preserved. This isolated branch contains only the requested Deck changes.

The wheel exposes Previous/Next, page count, arrow/Page Up/Page Down and scroll
navigation for both entry points. Existing centre cycling and slot dispatch are
preserved. The bar now routes clicks using its last successfully painted geometry
instead of recalculating time-dependent LIVE/pending chip geometry on input.
Resize, scene, visibility and failed-paint checks reject invalid painted boxes.
Current target availability remains authoritative when executing a control.

Code validation: clean baseline 180 Deck tests passed (`baseline-deck.log`);
candidate 211 Deck tests passed (`own-deck.log`), including navigation and expired
chip geometry regressions. Changed Python files pass Ruff and whitespace checks.
The new tests cover Preview/Search/Wheel as sibling lanes of the shared seam.

This PR does not include the separately reviewed display-mode cycle in PR #44.
No new UI/GPU tests were run: the owner requested code debugging without more UI
probing. These checks are not a full production/runtime/release certificate.
