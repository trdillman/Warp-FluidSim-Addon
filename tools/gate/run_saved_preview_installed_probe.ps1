param(
    [Parameter(Mandatory)][ValidateSet('Produce', 'Reopen')][string]$Stage,
    [Parameter(Mandatory)][string]$Blender,
    [Parameter(Mandatory)][string]$AddonModule,
    [Parameter(Mandatory)][string]$InstalledRoot,
    [Parameter(Mandatory)][string]$Archive,
    [Parameter(Mandatory)][string]$ProfileRoot,
    [Parameter(Mandatory)][string]$OutputRoot,
    [ValidateSet('surface', 'particles', 'cache-surface')][string]$Case = 'surface',
    [string]$InputBlend
)

$ErrorActionPreference = 'Stop'
if (-not $env:BGATE_LEASE_ID -or $env:BGATE_LEASE_CLASS -ne 'exclusive') {
    throw 'Run this probe inside the parent Blender Gate exclusive lease'
}
if ($env:CUDA_VISIBLE_DEVICES -eq '-1') { throw 'Parent GPU lane has CUDA disabled' }
if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) { throw 'Missing Blender executable' }
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw 'Missing installed extension archive' }
foreach ($path in @($ProfileRoot, $InstalledRoot)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Missing directory: $path" }
}
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$profilePath = (Resolve-Path -LiteralPath $ProfileRoot).Path
$installed = (Resolve-Path -LiteralPath $InstalledRoot).Path
$extensions = Join-Path $profilePath 'extensions'
if (-not $installed.StartsWith($extensions + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Installed root must belong to the supplied private extension repository'
}

$probe = @('--phase', $Stage.ToLowerInvariant(), '--case', $Case.ToLowerInvariant(), '--addon-module', $AddonModule,
           '--installed-root', $installed, '--source-root', $repo,
           '--archive', $Archive, '--output', $OutputRoot)
if ($Stage -eq 'Produce') {
    if (-not $InputBlend -or -not (Test-Path -LiteralPath $InputBlend -PathType Leaf)) {
        throw 'Produce requires an existing tsunami or dam-break input blend'
    }
    $probe += @('--input-blend', $InputBlend)
}

$profileEnvironment = @{
    BLENDER_USER_RESOURCES = $profilePath
    BLENDER_USER_CONFIG = Join-Path $profilePath 'config'
    BLENDER_USER_SCRIPTS = Join-Path $profilePath 'scripts'
    BLENDER_USER_EXTENSIONS = $extensions
    BLENDER_USER_DATAFILES = Join-Path $profilePath 'datafiles'
    PYTHONPATH = $null
    PYTHONHOME = $null
    PYTHONSTARTUP = $null
}
$previousEnvironment = @{}
foreach ($name in $profileEnvironment.Keys) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
}
try {
    foreach ($name in $profileEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $profileEnvironment[$name])
    }
    # Fresh processes share the private profile so Blender loads its saved extension repository.
    $probeScript = Join-Path $repo 'tests/acceptance/blender_saved_preview_acceptance.py'
    & $Blender --background --python-use-system-env --python-exit-code 1 --python $probeScript -- @probe
    if ($LASTEXITCODE) { throw "Installed save/reopen exit $LASTEXITCODE" }
}
finally {
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name])
    }
}
