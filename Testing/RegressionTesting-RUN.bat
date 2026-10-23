@echo off
setlocal
rem                                          testName         scene                   exposure  [camera]
call "%~dp0RegressionTesting-GENERATE.bat" bistro-cam0      bistro-programmer-art   4
call "%~dp0RegressionTesting-GENERATE.bat" bistro-cam1      bistro-programmer-art   5         "-7.685498,0.619501,4.641756,0.659771,-0.053973,0.749526,0.035664,0.998542,0.040515"
rem call "%~dp0RegressionTesting-GENERATE.bat" bistro-cam2      bistro-programmer-art   4         "-16.557644,3.978379,-2.673208,0.929426,0.077922,-0.360688,-0.072642,0.996960,0.028190"
call "%~dp0RegressionTesting-GENERATE.bat" bistro-cam3      bistro-programmer-art   4         "-14.142359,1.191548,6.900615,0.365480,-0.149435,-0.918746,0.055238,0.988771,-0.138858"
call "%~dp0RegressionTesting-GENERATE.bat" convergence-test convergence-test        0
call "%~dp0RegressionTesting-GENERATE.bat" kitchen          kitchen-with-test-stuff 2.0
call "%~dp0RegressionTesting-COMPARE.bat"
