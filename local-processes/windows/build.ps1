param(
    [Parameter(Mandatory = $true)][string] $ProfilePath,
    [Parameter(Mandatory = $true)][string] $ExpectedCommit,
    [string] $ScratchSerial,
    [string] $ScratchDriveLetter,
    [string] $RunId,
    [switch] $PreflightOnly
)

$ErrorActionPreference = "Stop"
$profile = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json
$sourceRoot = $profile.guest_repo_root
$artifactRoot = $profile.artifacts.guest_root
$disk = $null
$scratchMounted = $false
$scratchRoot = $null
$buildRoot = $null
$artifactRunDir = if ($RunId) { Join-Path (Join-Path $artifactRoot $ExpectedCommit.Substring(0, 12)) $RunId } else { $null }
$startedAt = (Get-Date).ToUniversalTime()

function Assert-SafeValue([string] $value, [string] $label) {
    if ($value -match '["&|<>^%]' -or $value.Contains([char]13) -or $value.Contains([char]10)) {
        throw "$label contains a command-shell metacharacter"
    }
}

Assert-SafeValue $profile.project "project name"
if ($ExpectedCommit -notmatch '^[0-9a-f]{40}$') {
    throw "ExpectedCommit must be a full lowercase Git commit hash"
}
if ($RunId -and $RunId -notmatch '^[A-Za-z0-9-]+$') {
    throw "RunId contains unsupported characters"
}
if ($ScratchSerial -and $ScratchSerial -notmatch '^[A-Za-z0-9]+$') {
    throw "ScratchSerial contains unsupported characters"
}

function Copy-BuildLogs {
    if ($PreflightOnly -or -not $artifactRunDir) {
        return
    }
    try {
        $logDir = Join-Path $artifactRunDir "logs"
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        if ($buildRoot) {
            foreach ($name in @("configure.log", "build.log", "runner.cmd")) {
                $path = Join-Path $buildRoot $name
                if (Test-Path -LiteralPath $path) {
                    Copy-Item -LiteralPath $path -Destination $logDir -Force
                }
            }
        }
    } catch {
        Write-Warning "Could not export build logs to $artifactRunDir : $_"
    }
}

try {
    if ($env:COMPUTERNAME -and $profile.guest_hostname -and
        $env:COMPUTERNAME -ine $profile.guest_hostname) {
        throw "Connected to $env:COMPUTERNAME, expected $($profile.guest_hostname)"
    }
    $activeBuildProcesses = @(Get-Process cmake,ninja,nvcc,cl -ErrorAction SilentlyContinue)
    if ($activeBuildProcesses.Count -gt 0) {
        throw "A build tool is already active on this VM: $($activeBuildProcesses.ProcessName -join ', ')"
    }

    $repoHead = (& git -C $sourceRoot rev-parse HEAD 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $repoHead -ne $ExpectedCommit) {
        throw "Mapped source HEAD '$repoHead' does not match expected commit '$ExpectedCommit'"
    }
    foreach ($required in @("CMakeLists.txt", "src/llama-kv-cache.cpp", "tools/server/server-prompt-cache-disk.cpp")) {
        if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot $required))) {
            throw "Mapped source is missing $required"
        }
    }

    $artifactDrive = (Split-Path -Qualifier $profile.artifacts.guest_root).TrimEnd(':')
    $artifactFree = (Get-PSDrive -Name $artifactDrive -ErrorAction Stop).Free
    if ($artifactFree -lt [int64]$profile.artifacts.minimum_free_bytes) {
        throw "Host artifact share has only $artifactFree free bytes"
    }

    $cuda = $profile.windows_build
    if ($cuda.cuda_path) {
        if (-not (Test-Path -LiteralPath $cuda.cuda_path)) {
            throw "Configured CUDA path is missing: $($cuda.cuda_path)"
        }
        $env:CUDA_PATH = $cuda.cuda_path
        $env:PATH = (Join-Path $cuda.cuda_path "bin") + ";" + $env:PATH
    }
    $nvcc = Get-Command nvcc.exe -ErrorAction Stop
    $nvccInfo = (& $nvcc.Source --version 2>&1 | Out-String)
    if ($cuda.cuda_version -and $nvccInfo -notmatch ("release " + [regex]::Escape($cuda.cuda_version))) {
        throw "nvcc version does not match profile CUDA version $($cuda.cuda_version): $nvccInfo"
    }
    Get-Command cmake.exe,ninja.exe -ErrorAction Stop | Out-Null
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "vswhere.exe not found at $vswhere"
    }
    $vsInstall = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1).Trim()
    $vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) {
        throw "vcvarsall.bat not found at $vcvars"
    }
    if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "The SSH session must be elevated to initialize the temporary scratch disk"
    }
    if ($PreflightOnly) {
        Write-Output "PREFLIGHT_OK host=$env:COMPUTERNAME commit=$repoHead cuda=$($cuda.cuda_version) artifact_free=$artifactFree"
        exit 0
    }
    if (-not $ScratchSerial -or -not $ScratchDriveLetter -or -not $RunId) {
        throw "A full build requires scratch serial, drive letter, and run ID"
    }

    $deadline = (Get-Date).AddSeconds(90)
    do {
        $matches = @(Get-Disk | Where-Object { ($_.SerialNumber -replace '\s', '').ToUpperInvariant() -eq $ScratchSerial.ToUpperInvariant() })
        if ($matches.Count -eq 1) {
            $disk = $matches[0]
            break
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    if (-not $disk) {
        throw "Could not find exactly one attached scratch disk with serial $ScratchSerial"
    }
    if ($disk.PartitionStyle -ne "RAW") {
        throw "Refusing to format scratch disk $ScratchSerial because its partition style is $($disk.PartitionStyle), not RAW"
    }
    if (Get-PSDrive -Name $ScratchDriveLetter -ErrorAction SilentlyContinue) {
        throw "Scratch drive letter $ScratchDriveLetter`: is already in use"
    }

    if ($disk.IsOffline) {
        Set-Disk -Number $disk.Number -IsOffline $false
    }
    if ($disk.IsReadOnly) {
        Set-Disk -Number $disk.Number -IsReadOnly $false
    }
    Initialize-Disk -Number $disk.Number -PartitionStyle GPT | Out-Null
    $partition = New-Partition -DiskNumber $disk.Number -UseMaximumSize -DriveLetter $ScratchDriveLetter
    Format-Volume -Partition $partition -FileSystem NTFS -NewFileSystemLabel "WINBUILD" -Confirm:$false -Force | Out-Null
    $scratchMounted = $true

    $scratchRoot = "$ScratchDriveLetter`:\local-processes"
    $buildRoot = Join-Path $scratchRoot (Join-Path $profile.project (Join-Path $ExpectedCommit.Substring(0, 12) $RunId))
    New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
    $env:TEMP = $scratchRoot
    $env:TMP = $scratchRoot
    $env:CUDA_CACHE_PATH = Join-Path $scratchRoot "cuda-cache"
    New-Item -ItemType Directory -Force -Path $env:CUDA_CACHE_PATH | Out-Null

    $nvcc = Get-Command nvcc.exe -ErrorAction Stop
    $nvccInfo = (& $nvcc.Source --version 2>&1 | Out-String)
    if ($cuda.cuda_version -and $nvccInfo -notmatch ("release " + [regex]::Escape($cuda.cuda_version))) {
        throw "nvcc version does not match profile CUDA version $($cuda.cuda_version): $nvccInfo"
    }
    $cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
    $ninja = (Get-Command ninja.exe -ErrorAction Stop).Source
    $cmakeOptions = @($cuda.cmake_options)
    $cmakeOptions += "-DCMAKE_CUDA_ARCHITECTURES=$($cuda.cuda_architectures)"
    $cmakeOptions += "-DCMAKE_BUILD_TYPE=$($cuda.configuration)"
    Assert-SafeValue $cuda.generator "CMake generator"
    foreach ($option in $cmakeOptions) {
        Assert-SafeValue $option "CMake option"
    }
    $cmakeSourceRoot = $sourceRoot.Replace('\', '/')
    Assert-SafeValue $cmakeSourceRoot "source root"
    Assert-SafeValue $buildRoot "build root"

    $buildDir = Join-Path $buildRoot "build"
    $configureLog = Join-Path $buildRoot "configure.log"
    $buildLog = Join-Path $buildRoot "build.log"
    $runnerCmd = Join-Path $buildRoot "runner.cmd"
    $quotedOptions = $cmakeOptions -join " "
    $targetArgs = if (@($cuda.targets).Count -gt 0) { "--target " + (@($cuda.targets) -join " ") } else { "" }
    $runnerLines = @(
        "@echo off",
        "call `"$vcvars`" x64 >nul 2>&1",
        "if errorlevel 1 exit /b 11",
        "`"$cmake`" -S `"$cmakeSourceRoot`" -B `"$buildDir`" -G `"$($cuda.generator)`" $quotedOptions > `"$configureLog`" 2>&1",
        "if errorlevel 1 exit /b 12",
        "`"$cmake`" --build `"$buildDir`" --config $($cuda.configuration) -j $([int]$cuda.jobs) $targetArgs > `"$buildLog`" 2>&1",
        "exit /b %ERRORLEVEL%"
    )
    Set-Content -LiteralPath $runnerCmd -Value $runnerLines -Encoding Ascii
    Write-Output "Configuring and building $($profile.project) at $ExpectedCommit using $($nvcc.Source)"
    $process = Start-Process -FilePath $env:ComSpec -ArgumentList @("/d", "/c", "`"$runnerCmd`"") -Wait -PassThru -NoNewWindow
    if ($process.ExitCode -ne 0) {
        Copy-BuildLogs
        throw "CMake/Ninja failed with exit code $($process.ExitCode); see $configureLog and $buildLog"
    }

    $binDir = Join-Path $buildDir ("bin\" + $cuda.configuration)
    $dlls = @(Get-ChildItem -LiteralPath $binDir -Filter "*.dll" -File -ErrorAction SilentlyContinue)
    $executables = @()
    foreach ($name in @($profile.artifacts.expected_executables)) {
        $path = Join-Path $binDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            Copy-BuildLogs
            throw "Expected build output was not produced: $path"
        }
        $executables += Get-Item -LiteralPath $path
    }
    $requiredBytes = ($dlls + $executables | Measure-Object -Property Length -Sum).Sum
    $freeBytes = (Get-PSDrive -Name $artifactDrive).Free
    if ($freeBytes -lt $requiredBytes + [int64]$profile.artifacts.minimum_free_bytes) {
        Copy-BuildLogs
        throw "Host artifact share has insufficient free space: $freeBytes bytes, need $($requiredBytes + [int64]$profile.artifacts.minimum_free_bytes)"
    }

    New-Item -ItemType Directory -Force -Path $artifactRunDir | Out-Null
    $binOut = Join-Path $artifactRunDir "bin"
    $logOut = Join-Path $artifactRunDir "logs"
    New-Item -ItemType Directory -Force -Path $binOut,$logOut | Out-Null
    foreach ($file in ($dlls + $executables)) {
        Copy-Item -LiteralPath $file.FullName -Destination $binOut -Force
    }
    Copy-BuildLogs
    $manifest = [ordered]@{
        project = $profile.project
        commit = $ExpectedCommit
        vm = $env:COMPUTERNAME
        cuda_version = $cuda.cuda_version
        cuda_architectures = $cuda.cuda_architectures
        configuration = $cuda.configuration
        targets = @($cuda.targets)
        started_utc = $startedAt.ToString("o")
        completed_utc = (Get-Date).ToUniversalTime().ToString("o")
        files = Get-ChildItem -LiteralPath $binOut -File | ForEach-Object { @{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } }
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $artifactRunDir "build-manifest.json") -Encoding UTF8
    Write-Output "BUILD_SUCCESS artifact=$artifactRunDir"
} catch {
    Copy-BuildLogs
    Write-Error $_
    exit 1
} finally {
    if ($scratchMounted -and $disk) {
        try {
            Set-Disk -Number $disk.Number -IsOffline $true -ErrorAction Stop
        } catch {
            Write-Warning "Could not offline scratch disk: $_"
        }
    }
}
