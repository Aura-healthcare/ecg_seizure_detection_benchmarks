from custom_features import *

rr_intervals_list_test = [
    739,711,731,701,895,729,709,716,767,779,890,748,821,702,831,829,840,817,703,866,804,744,899,792,720,737,805,792,775,704,735,
    742,821,882,896,803,864,790,842,772,828,799,808,717,833,701,826,721,784,886,727,767,894,846,894,855,832,853,798,770,754,892,811,705,761,830,874,878,883,852,869,854,750,899,
    895,824,717,879,811,833,864,731,761,899,736,898,835,771,740,783,
    0,828,768,727,736,837,721,739,779,703,866,772,875,756,752,852,870,798,880,829,883,763, 895,823,715, 900,
    ]

def test_filter_median():
    #La présence d'un 0 (manquement d'un battement) devrait obligatoirement modifier la liste ici.
    L1=len(rr_intervals_list_test)
    rr_intervals_list_filtered=filter_median(rr_intervals_list_test,7)
    L2= len(rr_intervals_list_filtered)
    assert L1==L2
    assert rr_intervals_list_filtered!=rr_intervals_list_test

def test_reg_Lin():
    x=[1,2,3]
    y=[1,2,3]
    a,b = reg_Lin(x,y)
    assert a==1

def test_slope():
    print(slope([1,2,3]))
    assert slope([1,2,3])== 1

def test_get_custom_features():
    assert get_custom_features(rr_intervals_list_test)!={}
    assert type(get_custom_features(rr_intervals_list_test))==type({})



if __name__=="__main__":
    test_filter_median()
    test_slope()
    test_reg_Lin()
    test_get_custom_features()