$ErrorActionPreference = "Stop"

function Ensure-WindowsCdmPatch {
    param(
        [Parameter(Mandatory = $true)][string]$Checkout,
        [Parameter(Mandatory = $true)][string]$Patch
    )

    git -C $Checkout apply --check $Patch
    if ($LASTEXITCODE -eq 0) {
        git -C $Checkout apply $Patch
        if ($LASTEXITCODE -ne 0) { throw "Windows CDM patch apply failed" }
        return
    }

    git -C $Checkout apply --reverse --check $Patch
    if ($LASTEXITCODE -ne 0) {
        throw "Windows CDM patch state is partial or corrupt"
    }
}

function Invoke-PrepareOmniDocBenchV16 {
    $Commit = "147cd5ac9472002f5751221d390bf00abdbc0d2f"
    $Checkout = "eval/.omnidocbench"
    $Patch = (Resolve-Path "eval/patches/omnidocbench-v16-windows-cdm.patch")

    if (-not (Test-Path "$Checkout/.git")) {
        git clone https://github.com/opendatalab/OmniDocBench.git $Checkout
        if ($LASTEXITCODE -ne 0) { throw "OmniDocBench clone failed" }
    }

    git -C $Checkout fetch origin $Commit
    if ($LASTEXITCODE -ne 0) { throw "OmniDocBench fetch failed" }

    git -C $Checkout checkout --detach $Commit
    if ($LASTEXITCODE -ne 0) { throw "OmniDocBench checkout failed" }

    Ensure-WindowsCdmPatch -Checkout $Checkout -Patch $Patch

    python eval/benchmark_contract.py --checkout $Checkout
    if ($LASTEXITCODE -ne 0) { throw "v1.6 contract validation failed" }
}

if ($MyInvocation.InvocationName -ne ".") {
    Invoke-PrepareOmniDocBenchV16
}
