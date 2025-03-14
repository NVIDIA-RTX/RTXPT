
$scene_file=$args[0]
$sample_count=$args[1]
$output_file_bmp=$args[2]

Write-Verbose "Running $scene_file using $sample_count samples... "

# make sure output path exists
$file_dir = Split-Path -Path $output_file_bmp;
if(!(Test-Path -PathType container $file_dir))
{
	$res = New-Item -ItemType Directory -Path $file_dir
}

$test_reference = "..\bin\pt_sdk.exe --scene $scene_file --nonInteractive --noStreamline --noWindow --width 960 --height 540 --screenshotFrameIndex $sample_count --screenshotFileName $output_file_bmp"
cmd /c $test_reference > output/$scene_file.$sample_count.log.txt

if (! $?)
{
	Write-Error "Failed to generate screenshot exit code. Application returned error $LASTEXITCODE"
	exit 1
}
else
{
	Write-Verbose "Screenshot successfully generated at $pwd\output\$scene_file.$sample_count.bmp"
	
}