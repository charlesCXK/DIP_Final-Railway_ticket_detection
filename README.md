# DIP-Final
### I. Install requirements

```shell
$ ./install-requirement.sh
```

### II. How to test it

- 假设包含测试图片的文件夹为 test_data
- 假设包含测试图片文件名的文本文件为 annotation.txt
- 将以上两个文件放入 code/ 文件夹中

```Shell
$ cd code
$ python3 evaluate.py --dir test_data --txt annotation.txt
```

