from sdepy import integrate


@integrate
def my_process(t, x, theta=1.0, k=1.0, sigma=1.0):
    return {"dt": k * (theta - x), "dw": sigma}


print(my_process())
