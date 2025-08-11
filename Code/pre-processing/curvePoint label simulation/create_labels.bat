setlocal enableextensions enabledelayedexpansion
set procdirec=C:\Program Files\processing-4.3-windows-x64\processing-4.3\
set sketchdirec=C:\Users\20183777\Desktop\simulation\simulation\
set outpdirec=C:\Users\20183777\Desktop\simulation\output\
set labeldirec=C:\Users\20183777\Desktop\simulation\labels\
set scenarios=1 2
set experiments=
set experiments1=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
set experiments2=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108

cd /d %procdirec%
@echo on
rmdir %outpdirec% /s /q
@echo on
rmdir %labeldirec% /s /q
@echo on
md %labeldirec%
@echo on

for %%i in (%scenarios%) do (
   if %%i==1 (set experiments=%experiments1%) else (set experiments=%experiments2%)
   for %%j in (!experiments!) do (
      start /wait processing-java.exe --sketch=%sketchdirec% --output=%outpdirec% --force --run -sc %%i -exp %%j -out %labeldirec%
	  @echo on
   )
)
