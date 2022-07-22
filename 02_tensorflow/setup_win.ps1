
Write-Output("To activate this environment, use")
Write-Output("$ conda activate tensorflow")
Write-Output("To deactivate an active environment, use")
Write-Output("$ conda deactivate")

conda create -n tensorflow pip python=3.9

Write-Output("setup complete")