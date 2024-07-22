import nestbox

if __name__ == '__main__':
    nestbox.create_coordinate_system('cs1')
    cs1_point = [1, 2, 3]
    cs2_point = nestbox.from_cs('cs1').to_cs('cs2').transform(cs1_point)
    print(f'Point in cs1: {cs1_point}. Point in cs2: {cs2_point}')
