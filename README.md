# dicom-database

>https://hackmd.io/LgoBKB6WTJqvOruG_7rkjw?both

Base on pydicom to develope an analyze platform to analyze the Dicom

dicom_read 內有方法的使用
dicommethodcore 內有全部的方法
h rceord =id +Study Date +Study Instance UID

> Dicom relation connect
> MR 影像種類 看series description

    1. AX1:T1 AX
    2. AX2:T2 AX (see tumor apparently)
    3. COR:T1 Cor (橫切面)
    4. AXC:T1 AX 細緻的影像
    5. AXCC 打藥的影像，resolution 低
    6. AXS:orher AX
    7. TOF:去頭骨影像
    
datarange
> Store: row major

先存row 在存column =>所以如果x see column as y see row
所以如果要extract data 中[a,b] 應該抓data[b,a] 
ex: if want extract  a 中[281:290,300:400]
=>a[300:400,281:290]

>  DOSE

    1. dose max: pixel max value*dosegrid (Unit:GY)
    2. Dosegrid: 格狀衰變數值 通常corridinate 橫跨越多，值越低
    3. 檔案數量： Skull+ (rectangle dosgrid+ circle tumor)*(tumor)

> DICOM 座標

    1. X:right + left-
    2. Y:fornt + back -
    3. Z:up- down+
    4. 這是相對座標 X Y 只是剛好在每個系列會對到

# Requirements
* opencv
* opencv-contrib-python
* numpy 
* math
* scipy
* skscikit-image
* matplotlib
* scikit-learn
* plotly
* glob
* random
* dicompylercore 
* six 
* shapely 
* pydicom 

To install all the requirements run
```
pip install -r requirements.txt
```
