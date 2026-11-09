param(
  [switch]$SkipCoco,
  [switch]$SkipOpenImages
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$dataRoot = Join-Path $projectRoot 'datasets'
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw "Python environment not found: $pythonExe. Create .venv and install requirements first."
}

function Get-File([string]$Url, [string]$Destination) {
  if (Test-Path -LiteralPath $Destination) {
    Write-Host "Already downloaded: $Destination"
    return
  }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
  $partialPath = "$Destination.part"
  Write-Host "Downloading $Url"
  & curl.exe --location --fail --retry 3 --retry-delay 2 --continue-at - --output $partialPath $Url
  if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
  Move-Item -LiteralPath $partialPath -Destination $Destination -Force
}

function Expand-Once([string]$Archive, [string]$Destination, [string]$ExpectedPath) {
  if (Test-Path -LiteralPath $ExpectedPath) {
    Write-Host "Already extracted: $ExpectedPath"
    return
  }
  Write-Host "Extracting $Archive"
  Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
}

if (-not $SkipCoco) {
  $cocoRoot = Join-Path $dataRoot 'coco'
  $valZip = Join-Path $cocoRoot 'val2017.zip'
  $annotationsZip = Join-Path $cocoRoot 'annotations_trainval2017.zip'
  Get-File 'https://images.cocodataset.org/zips/val2017.zip' $valZip
  Get-File 'https://images.cocodataset.org/annotations/annotations_trainval2017.zip' $annotationsZip
  Expand-Once $valZip $cocoRoot (Join-Path $cocoRoot 'val2017')
  Expand-Once $annotationsZip $cocoRoot (Join-Path $cocoRoot 'annotations\instances_val2017.json')

  $prepareCoco = Join-Path $projectRoot 'prepare_coco.py'
  & $pythonExe $prepareCoco --images (Join-Path $cocoRoot 'val2017') --annotations (Join-Path $cocoRoot 'annotations\instances_val2017.json') --out (Join-Path $cocoRoot 'full_val2017') --num-images 5000 --seed 42
  if ($LASTEXITCODE -ne 0) { throw 'Preparing full COCO val2017 failed.' }
  & $pythonExe $prepareCoco --images (Join-Path $cocoRoot 'val2017') --annotations (Join-Path $cocoRoot 'annotations\instances_val2017.json') --out (Join-Path $cocoRoot 'subset_500') --num-images 500 --seed 42
  if ($LASTEXITCODE -ne 0) { throw 'Preparing the first COCO subset failed.' }
  & $pythonExe $prepareCoco --images (Join-Path $cocoRoot 'val2017') --annotations (Join-Path $cocoRoot 'annotations\instances_val2017.json') --out (Join-Path $cocoRoot 'subset_500_image_set2') --num-images 500 --seed 1043 --exclude-manifest (Join-Path $cocoRoot 'subset_500\manifest.json')
  if ($LASTEXITCODE -ne 0) { throw 'Preparing the second COCO subset failed.' }
  & $pythonExe $prepareCoco --images (Join-Path $cocoRoot 'val2017') --annotations (Join-Path $cocoRoot 'annotations\instances_val2017.json') --out (Join-Path $cocoRoot 'subset_500_image_set3') --num-images 500 --seed 1044 --exclude-manifest (Join-Path $cocoRoot 'subset_500\manifest.json') --exclude-manifest (Join-Path $cocoRoot 'subset_500_image_set2\manifest.json')
  if ($LASTEXITCODE -ne 0) { throw 'Preparing the third COCO subset failed.' }
}

if (-not $SkipOpenImages) {
  $sourceRoot = Join-Path $dataRoot 'open_images_v7\source'
  $sourceFiles = @{
    'oidv7-class-descriptions-boxable.csv' = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'
    'validation-annotations-bbox.csv' = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
    'validation-images-with-rotation.csv' = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
  }
  foreach ($file in $sourceFiles.GetEnumerator()) {
    Get-File $file.Value (Join-Path $sourceRoot $file.Key)
  }

  $subsetDefinitions = @(
    @{ Root = (Join-Path $dataRoot 'open_images_v7'); Ids = 'open_images_100_image_ids.txt'; Cap = '100' },
    @{ Root = (Join-Path $dataRoot 'open_images_v7_test_extra'); Ids = 'open_images_300_image_ids.txt'; Cap = '300' }
  )
  foreach ($subset in $subsetDefinitions) {
    New-Item -ItemType Directory -Force -Path $subset.Root | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot "benchmark_data\$($subset.Ids)") -Destination (Join-Path $subset.Root 'image_ids.txt') -Force
    & $pythonExe (Join-Path $projectRoot 'prepare_open_images_subset.py') --root $subset.Root --source-root $sourceRoot --max-images $subset.Cap
    if ($LASTEXITCODE -ne 0) { throw "Preparing Open Images subset failed: $($subset.Root)" }
    & $pythonExe (Join-Path $projectRoot 'download_open_images_subset.py') --root $subset.Root --workers 5
    if ($LASTEXITCODE -ne 0) { throw "Downloading Open Images subset failed: $($subset.Root)" }
  }
}

Write-Host 'Project evaluation datasets are ready under datasets/.' -ForegroundColor Green
Write-Host 'Large archives, extracted images, and generated labels remain local and are not added to Git.'
