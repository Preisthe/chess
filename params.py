size = (95, 95)

mark = [
    'b_paw',
    'b_can',
    'b_can',
    'r_ele',
    'b_ele',
    'r_can',
    'b_man',
    'r_paw',
    'b_ele',
    'r_kin',
    'b_roo',
    'b_paw',
    'b_kin',
    'r_ele',
    'b_man',
    'r_hor',
    'b_hor',
    'r_paw',
    'r_paw',
    'r_man',
    'r_can',
    'r_paw',
    'b_paw',
    'r_man',
    'r_roo',
    'r_hor',
    'b_hor',
    'b_roo',
    'r_roo',
    'b_paw',
    'b_paw',
    'r_paw'
]

num_classes = 14

types = [
    'r_kin',
    'r_man',
    'r_ele',
    'r_hor',
    'r_roo',
    'r_paw',
    'r_can',
    'b_kin',
    'b_man',
    'b_ele',
    'b_hor',
    'b_roo',
    'b_paw',
    'b_can'
]

chinese = [
    '红帅',
    '红士',
    '红相',
    '红马',
    '红车',
    '红兵',
    '红炮',
    '黑将',
    '黑士',
    '黑象',
    '黑马',
    '黑车',
    '黑卒',
    '黑炮'
]

duplicates = {
    'r_kin': 1,
    'r_man': 2,
    'r_ele': 2,
    'r_hor': 2,
    'r_roo': 2,
    'r_paw': 5,
    'r_can': 2,
    'b_kin': 1,
    'b_man': 2,
    'b_ele': 2,
    'b_hor': 2,
    'b_roo': 2,
    'b_paw': 5,
    'b_can': 2
}

vali = [
    'r_man', 'b_man', 'r_man', 'r_ele', 'b_ele', 'r_ele',
    'b_ele', 'b_kin', 'r_kin', 'b_man', 'r_ele', 'r_ele',
    'b_kin', 'b_man', 'b_man', 'r_kin', 'b_ele', 'b_ele',
    'r_man', 'r_man', 'r_ele', 'r_man', 'r_man', 'b_ele',
    'b_man', 'b_man', 'r_ele', 'b_kin', 'r_kin', 'b_ele',
    'r_hor', 'r_hor', 'b_roo', 'b_hor', 'b_hor', 'r_roo',
    'b_roo', 'r_roo', 'b_roo', 'r_roo', 'b_hor', 'r_hor',
    'r_roo', 'b_roo', 'b_hor', 'r_hor', 'b_roo', 'b_roo',
    'b_hor', 'r_hor', 'r_hor', 'r_roo', 'r_roo', 'b_hor',
    'r_paw', 'b_paw', 'b_can', 'r_paw', 'r_paw', 'b_paw',
    'b_paw', 'b_paw', 'r_paw', 'r_can', 'r_paw', 'r_can',
    'b_paw', 'b_can', 'b_paw', 'b_paw', 'r_paw', 'r_paw',
    'b_paw', 'b_can', 'r_can', 'b_paw', 'r_can', 'b_can',
    'r_paw', 'r_paw', 'b_paw', 'r_paw', 'r_can', 'r_can',
    'b_paw', 'r_paw', 'b_can', 'r_paw', 'r_paw', 'b_paw',
    'r_paw', 'b_paw', 'b_can', 'r_paw', 'b_paw', 'b_paw'
]