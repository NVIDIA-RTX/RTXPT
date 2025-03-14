$golden_image=$args[0]
$threshold=$args[1]
$scene_file=$args[2]
$sample_count=$args[3]
$metric=$args[4]

# Make sure output folder exists
$output_dir = "$pwd\output"
if(!(Test-Path -PathType container $output_dir))
{
	New-Item -ItemType Directory -Path $output_dir
}

$output_file = "$output_dir\$scene_file.$sample_count.bmp"

# Generate screenshot
# Render image to $output_file
.\_1_render.ps1 $scene_file $sample_count $output_file

if ($LASTEXITCODE -ne 0)
{
	exit $LASTEXITCODE
}

# Compare screenshot against golden image
$output_psnr_dir = "$pwd\output\$scene_file.$sample_count.$metric.bmp"
.\_2_compare.ps1 $golden_image $output_file $output_psnr_dir $threshold $metric

exit $LASTEXITCODE