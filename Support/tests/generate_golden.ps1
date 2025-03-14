# ----------------------------------------------------

$testConfig = Get-Content "$pwd/tests.json" | ConvertFrom-Json

# Make sure output folder exists
$outputDir = "$pwd\golden_image"

if(!(Test-Path -PathType container $outputDir))
{
	New-Item -ItemType Directory -Path $outputDir
}

foreach ($test in $testConfig.Tests) {
	
	$sample_count = $($test.sampleCount)
	$input_scene = "$($test.Scene).scene.json"
	$output_file_bmp = "$outputDir\$($test.Scene).$sample_count.bmp"

	Write-Host "$input_scene -> $output_file_bmp"

	.\_1_render.ps1 $input_scene $sample_count $output_file_bmp

	if ($LASTEXITCODE -ne 0)
	{
		exit $LASTEXITCODE
	}
}