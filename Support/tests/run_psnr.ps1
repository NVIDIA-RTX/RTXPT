function GeneratePSNR
{
	Param ($scene, $sampleCount, $threshold, $metric)

	Write-Host "$threshold $scene $sampleCount $metric"

	$info = ./_run.ps1 $pwd\golden_image\$scene.$sampleCount.bmp $threshold $scene $sampleCount $metric 6>&1

	if ($LASTEXITCODE -eq 1) # Test fail - Error too large
	{
		Write-Host "ERROR $scene.$sampleCount.bmp".PadRight(50, ' ') -ForegroundColor Red -nonewline 
		Write-Host " [$info]"
		return 0
	}

	Write-Host "$pwd\golden_image\$scene.$sampleCount.bmp"

	Write-Host "      $info"
	return 1
}

$iterations = 1

if ($args[0])
{
	$iterations = $args[0];
}

# ----------------------------------------------------

$jsonFile = "tests.json"
$testConfig = Get-Content "$pwd/$jsonFile" | ConvertFrom-Json

$pass = [int]0
$completed = [int]0

foreach ($test in $testConfig.Tests) {

	if (1 -eq (GeneratePSNR -scene $($test.Scene) -sampleCount $($test.sampleCount) -threshold $($test.threshold) -metric "PSNR"))
	{
		$pass = $pass + 1
	}
	$completed = $completed + 1
}
