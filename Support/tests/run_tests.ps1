function RunTestScene
{
	Param ($scene, $sampleCount, $threshold)

	Write-Host "[RUN       ] $scene.$sampleCount.bmp" -ForegroundColor Green

	$info = ./_run.ps1 $pwd\golden_image\$scene.$sampleCount.bmp $threshold $scene $sampleCount "AE" 6>&1

	if ($LASTEXITCODE -eq 0) # Success
	{
		Write-Host "[       OK ] $scene.$sampleCount.bmp".PadRight(50, ' ') -ForegroundColor Green -nonewline
		Write-Host " [$info]"
		return 1
	}
	elseif ($LASTEXITCODE -eq 1) # Test fail - Error too large
	{
		Write-Host "[  FAILED  ] $scene.$sampleCount.bmp".PadRight(50, ' ') -ForegroundColor Red -nonewline 
		Write-Host " [$info]"
		return 0
	}
	else # Other error / app or internal 
	{
		Write-Host "[  FAILED  ] $scene.$sampleCount.bmp" -ForegroundColor Red
		Write-Host "      $info"
		return 0
	}
}

$iterations = 1

if ($args[0])
{
	$iterations = $args[0];
}

# ----------------------------------------------------

$jsonFile = "tests.json"
$testConfig = Get-Content "$pwd/$jsonFile" | ConvertFrom-Json

Write-Host "$iterations x @ $jsonFile"

$pass = [int]0
$completed = [int]0

for ($i=1; $i -le $iterations; $i++) {

	foreach ($test in $testConfig.Tests) {

		# Write the progress bar
		# $totalTests = $iterations * [int]$testConfig.Tests.Length
		# $percentage = 100 * [decimal]$completed/[decimal]$totalTests
		# Write-Progress -Id 1 -PercentComplete $percentage -Status "Processing" -Activity "$completed out of $totalTests tests completed"

		if (1 -eq (RunTestScene -scene $($test.Scene) -sampleCount $($test.sampleCount) -threshold $($test.threshold)))
		{
			$pass = $pass + 1
		}
		$completed = $completed + 1
	}
}

# Return error if any test failed
if ($pass -eq $completed)
{
	Write-Host "[  PASSED  ] $pass tests" -ForegroundColor Green
	exit (0)
}
else
{
	$failed = $completed-$pass
	Write-Host "$failed FAILED TESTS" -ForegroundColor Red
	exit (1)
}