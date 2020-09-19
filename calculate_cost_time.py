def calculate_time(package, length, band_width):
    time1 = (length * 1000) / (3 * pow(10, 8))
    time2 = package / (band_width * pow(10, 6))
    return (time1 + time2) * (pow(10, 3))


if __name__ == '__main__':
    time_l1 = calculate_time(12000, 3, 100)
    time_l3 = calculate_time(12000, 1000, 10)
    time_l5 = calculate_time(12000, 3, 1000)
    print(time_l1)
    print(time_l3)
    print(time_l5)
    time = 2*time_l1 + time_l3 + time_l5
    result = round(time * 100) / 100
    print("%.1f" % result)