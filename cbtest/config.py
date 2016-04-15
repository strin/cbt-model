from common import *

if 'datadir' in os.environ:
    cbt_datadir = os.environ['datadir']
else:
    cbt_datadir = '../data/CBTest/data/'

cbt_cn_train = os.path.join(cbt_datadir, 'cbtest_CN_train.txt')
cbt_cn_test = os.path.join(cbt_datadir, 'cbtest_CN_test_2500ex.txt')
cbt_cn_val = os.path.join(cbt_datadir, 'cbtest_CN_valid_2000ex.txt')

cbt_ne_train = os.path.join(cbt_datadir, 'cbtest_NE_train.txt')
cbt_ne_test = os.path.join(cbt_datadir, 'cbtest_NE_test_2500ex.txt')
cbt_ne_val = os.path.join(cbt_datadir, 'cbtest_NE_valid_2000ex.txt')

cbt_p_train = os.path.join(cbt_datadir, 'cbtest_P_train.txt')
cbt_p_test = os.path.join(cbt_datadir, 'cbtest_P_test_2500ex.txt')
cbt_p_val = os.path.join(cbt_datadir, 'cbtest_P_valid_2000ex.txt')

cbt_v_train = os.path.join(cbt_datadir, 'cbtest_V_train.txt')
cbt_v_test = os.path.join(cbt_datadir, 'cbtest_V_test_2500ex.txt')
cbt_v_val = os.path.join(cbt_datadir, 'cbtest_V_valid_2000ex.txt')
