# NAS-HR
A heart rate estimation approach to automatically search a lightweight network that can achieve even higher accuracy than a complex network while reducing the computational cost.

----------------------------------------------------------------------------------------
****************************************************************************************
----------------------------------------------------------------------------------------
File "POS_STMap" give the calculation process of POS-STMap,

File "Search&Train" give the searching and training process.

----------------------------------------------------------------------------------------
****************************************************************************************
----------------------------------------------------------------------------------------
How to Run?

1. Get VIPL-HR from http://vipl.ict.ac.cn/view_database.php?id=15.

2. Get face landmarks refer to https://github.com/seetafaceengine/SeetaFace2 (any 81 landmarks is ok).

3. Get "POS_STMap" by the code "VIPL_Processing" in File "POS_STMap".

4. Search a efficient and lightweight CNN by the code "search.py" in File "Search&Train".

5. Train the searched CNN by the code "augment.py" in File "Search&Train".
----------------------------------------------------------------------------------------
****************************************************************************************
----------------------------------------------------------------------------------------

The part of NAS is based on DARTS framework from https://github.com/khanrc/pt.darts.

The part of POS is based on iphy form https://github.com/danmcduff/iphys-toolbox.

If you have any question, please concate with me: hao.lu@miracle.ict.ac.cn.
