@ECHO OFF
:: This batch file will upload files to Github automatically
TITLE GitPush

:: Section 1 - comment

ECHO ..............................

ECHO === 1/3 ===
git add *

ECHO ..............................

ECHO === 2/3 ===
git commit -am "-"

ECHO ..............................

ECHO === 3/3 ===
git push

ECHO  bye...
ECHO ..............................

exit /B
