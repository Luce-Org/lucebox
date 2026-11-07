param(
    [ValidateSet('Start', 'Stop', 'Restart', 'Status', 'Wait')]
    [string] $Command = 'Start',

    [string] $Distro = '',
    [string] $RepoPath = '/mnt/c/Users/adyba/src/lucebox-hub',
    [int] $WaitSeconds = 300,
    [int] $ContextSize = 70080,
    [int] $DraftContextSize = 2048,
    [int] $DraftNMax = 4,
    [int] $DraftBlockSize = 4,
    [int] $BatchSize = 2048,
    [int] $UBatchSize = 512,
    [string] $Model = '',
    [string] $CacheTypeK = 'turbo4',
    [string] $CacheTypeV = 'turbo4',
    [string] $DraftCacheTypeK = '',
    [string] $DraftCacheTypeV = '',
    [ValidateSet('atomic', 'llama-cpp', 'llama_cpp', 'spec-draft')]
    [string] $MtpStyle = 'atomic',
    [string] $LlamaServer = '',
    [string] $MtpModel = '',
    [string] $GpuLayersDraft = 'all',
    [string] $DraftOverrideTensor = 'token_embd.weight=CUDA0',
    [string] $CacheRam = '0',
    [switch] $NoKvOffload,
    [int] $Poll = 100,
    [ValidateSet(0, 1)]
    [int] $PollBatch = 1,
    [ValidateSet(-1, 0, 1, 2, 3)]
    [int] $Priority = 2,
    [ValidateSet(0, 1, 2, 3)]
    [int] $PriorityBatch = 2,
    [int] $ThreadsHttp = 1,
    [int] $GpuClockMin = 2100,
    [int] $GpuClockMax = 2700,
    [switch] $SkipGpuClockLock
)

$ErrorActionPreference = 'Stop'

$scriptPath = "$RepoPath/scripts/lucebox-gemma4-4090.sh"
$wslArgsPrefix = @()
if ($Distro -ne '') {
    $wslArgsPrefix += @('-d', $Distro)
}

function Invoke-LuceboxWsl {
    param([string] $Bash)
    & wsl.exe @wslArgsPrefix -e bash -lc $Bash
}

function New-WslArgumentLine {
    param([string] $Bash)

    $parts = @()
    $parts += $wslArgsPrefix
    $parts += @('-e', 'bash', '-lc', $Bash)

    ($parts | ForEach-Object {
        $part = [string] $_
        if ($part -match '[\s"]') {
            '"' + ($part -replace '"', '\"') + '"'
        } else {
            $part
        }
    }) -join ' '
}

$effectiveDraftCacheTypeK = if ($DraftCacheTypeK -ne '') { $DraftCacheTypeK } else { $CacheTypeK }
$effectiveDraftCacheTypeV = if ($DraftCacheTypeV -ne '') { $DraftCacheTypeV } else { $CacheTypeV }

function ConvertTo-BashSingleQuoted {
    param([string] $Value)
    $singleQuote = [char]39
    $singleQuote + ($Value -replace $singleQuote, ($singleQuote + '"' + $singleQuote + '"' + $singleQuote)) + $singleQuote
}

function Get-LuceboxEnvPrefix {
    $pairs = [ordered] @{
        LUCEBOX_GEMMA4_CTX_SIZE = [string] $ContextSize
        LUCEBOX_GEMMA4_DRAFT_CTX_SIZE = [string] $DraftContextSize
        LUCEBOX_GEMMA4_DRAFT_N_MAX = [string] $DraftNMax
        LUCEBOX_GEMMA4_DRAFT_BLOCK_SIZE = [string] $DraftBlockSize
        LUCEBOX_GEMMA4_BATCH_SIZE = [string] $BatchSize
        LUCEBOX_GEMMA4_UBATCH_SIZE = [string] $UBatchSize
        LUCEBOX_GEMMA4_CACHE_TYPE_K = $CacheTypeK
        LUCEBOX_GEMMA4_CACHE_TYPE_V = $CacheTypeV
        LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_K = $effectiveDraftCacheTypeK
        LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_V = $effectiveDraftCacheTypeV
        LUCEBOX_GEMMA4_MTP_STYLE = $MtpStyle
        LUCEBOX_GEMMA4_GPU_LAYERS_DRAFT = $GpuLayersDraft
        LUCEBOX_GEMMA4_DRAFT_OVERRIDE_TENSOR = $DraftOverrideTensor
        LUCEBOX_GEMMA4_CACHE_RAM = $CacheRam
        LUCEBOX_GEMMA4_NO_KV_OFFLOAD = if ($NoKvOffload) { '1' } else { '0' }
        LUCEBOX_GEMMA4_POLL = [string] $Poll
        LUCEBOX_GEMMA4_POLL_BATCH = [string] $PollBatch
        LUCEBOX_GEMMA4_PRIORITY = [string] $Priority
        LUCEBOX_GEMMA4_PRIORITY_BATCH = [string] $PriorityBatch
        LUCEBOX_GEMMA4_THREADS_HTTP = [string] $ThreadsHttp
    }
    if ($LlamaServer -ne '') {
        $pairs.LUCEBOX_LLAMA_SERVER = $LlamaServer
    }
    if ($Model -ne '') {
        $pairs.LUCEBOX_GEMMA4_MODEL = $Model
    }
    if ($MtpModel -ne '') {
        $pairs.LUCEBOX_GEMMA4_MTP_MODEL = $MtpModel
    }
    ($pairs.GetEnumerator() | ForEach-Object {
        "$($_.Key)=$(ConvertTo-BashSingleQuoted ([string] $_.Value))"
    }) -join ' '
}

$envPrefix = Get-LuceboxEnvPrefix

function Invoke-NvidiaSmi {
    param([string[]] $Arguments)

    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warning 'nvidia-smi.exe was not found; GPU clock control is skipped.'
        return
    }

    try {
        & $nvidiaSmi.Source @Arguments | Out-String | Write-Verbose
    } catch {
        Write-Warning "nvidia-smi.exe $($Arguments -join ' ') failed: $($_.Exception.Message)"
    }
}

function Set-LuceboxGpuClockLock {
    if ($SkipGpuClockLock) {
        return
    }
    Invoke-NvidiaSmi @('-lgc', "$GpuClockMin,$GpuClockMax")
}

function Reset-LuceboxGpuClockLock {
    if ($SkipGpuClockLock) {
        return
    }
    Invoke-NvidiaSmi @('-rgc')
}

switch ($Command) {
    'Start' {
        Set-LuceboxGpuClockLock
        Invoke-LuceboxWsl "rm -f `"`$HOME/lucebox-runs/lucebox-gemma4-mtp-server.pid`""
        $bash = "chmod +x '$scriptPath'; $envPrefix exec '$scriptPath' run"
        $startArgs = New-WslArgumentLine $bash
        $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $startArgs -PassThru -WindowStyle Hidden
        "winpid=$($proc.Id)"
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' wait $WaitSeconds"
    }
    'Stop' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' stop"
        Reset-LuceboxGpuClockLock
    }
    'Restart' {
        Set-LuceboxGpuClockLock
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' stop || true"
        $bash = "chmod +x '$scriptPath'; $envPrefix exec '$scriptPath' run"
        $startArgs = New-WslArgumentLine $bash
        $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList $startArgs -PassThru -WindowStyle Hidden
        "winpid=$($proc.Id)"
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' wait $WaitSeconds"
    }
    'Status' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' status"
    }
    'Wait' {
        Invoke-LuceboxWsl "chmod +x '$scriptPath'; $envPrefix '$scriptPath' wait $WaitSeconds"
    }
}
