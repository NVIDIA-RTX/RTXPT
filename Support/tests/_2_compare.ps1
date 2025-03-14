$golden_image=$args[0]
$target=$args[1]
$output=$args[2]
$threshold=$args[3]
$metric=$args[4]

write-verbose "Computing absolute error between golden image ($golden_image) and target ($target) threshold value $threshold to $output"

# magick writes the result via stderr ugh
# $metric = "AE"
# $metric = "PSNR"

if (!(Test-Path -Path $golden_image))
{
	Write-Error "Internal error: file ($golden_image) does not exist"
	exit 2
}

if (!(Test-Path -Path $target))
{
	Write-Error "Internal error: file ($target) does not exist"
	exit 2
}

$magick_output = &.\..\tools\ImageMagick\magick.exe compare -metric $metric $golden_image $target $output 2>&1

$splitOutput = $magick_output -split ' '

if ($splitOutput.Length -ne 0) {

	# PSNR result is written as X (Y), e.g 16.8849 (0.168849)
	# AE result is written as X, e.g 245 or scientific notation 1.31948e+06
	# the line below will deal with multiple words and scientific notation
	$image_delta = [double]::Parse($splitOutput[0], [Globalization.NumberStyles]::Float)

	if ([double]$image_delta -le [double]$threshold)
	{
		write-Information "Diff(=$image_delta $metric) is below threshold(=$threshold $metric)"

		write-Verbose "SUCCESS: $metric result $image_delta is below threshold ($threshold)"
		exit 0
	}
	else
	{
		write-Information "Diff(=$image_delta $metric) is above threshold(=$threshold $metric)"

		exit 1
	}
} else {
    Write-Error "Internal error: No number found in the string."
	exit 2
}