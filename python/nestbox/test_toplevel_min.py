import nestbox

cs1_point = [2, 3, 4]
cs2_point = nestbox.from_cs('cs1').to_cs('cs2').convert(cs1_point)
print(f'Point in cs1: {cs1_point}. Point in cs2: {cs2_point}')
