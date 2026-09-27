@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64 >nul 2>&1
cd /d D:\code\opencv_3\opencv_1\CH

echo ============ Compile WaveletTransform2D.cpp ============
cl /nologo /EHsc /W3 /I"D:\download\opencv\build\include" /c WaveletTransform2D.cpp
if errorlevel 1 goto :fail

echo ============ Compile MorphologicalHoleFillerComplete.cpp ============
cl /nologo /EHsc /W3 /I"D:\download\opencv\build\include" /c MorphologicalHoleFillerComplete.cpp
if errorlevel 1 goto :fail

echo ============ Compile CenterlineExtractor.cpp ============
cl /nologo /EHsc /W3 /I"D:\download\opencv\build\include" /c CenterlineExtractor.cpp
if errorlevel 1 goto :fail

echo ============ Compile ImageProcessor.cpp ============
cl /nologo /EHsc /W3 /I"D:\download\opencv\build\include" /c ImageProcessor.cpp
if errorlevel 1 goto :fail

echo ============ Compile main.cpp ============
cl /nologo /EHsc /W3 /I"D:\download\opencv\build\include" /c main.cpp
if errorlevel 1 goto :fail

echo ============ Compile LineEq.cpp ============
cl /nologo /EHsc /W3 /I"D:\download\opencv\build\include" /c LineEq.cpp
if errorlevel 1 goto :fail

echo ============ Link ============
link /nologo /OUT:opencv_3.exe WaveletTransform2D.obj MorphologicalHoleFillerComplete.obj CenterlineExtractor.obj ImageProcessor.obj main.obj LineEq.obj /LIBPATH:"D:\download\opencv\build\x64\vc14\lib" opencv_world460.lib
if errorlevel 1 goto :fail

echo.
echo ============ BUILD SUCCESS ============
goto :eof

:fail
echo.
echo ============ BUILD FAILED ============
exit /b 1
