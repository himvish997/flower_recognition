def print_function(metrics, time=None, lr=None, is_progress=False, step=1, total_steps=100):
    '''
    Display function for the training progress
    :param metrics: (dict) metrics which you want to display during the training
    :param time_: (int) time in seconds
    :param lr: (float) learning rate of that epoch (default is None)
    :param is_progress: (bool) True if want to display the progress bar(default is False)
    :param step: (int) nth steps
    :param total_steps: (int) total steps in which this function is called
    :return:
    '''
    if is_progress:
        progress_length = 30
        i = int(step * progress_length / total_steps)
        if progress_length == i:
            progress_bar = "[" + i*'=' + '=' + (progress_length - i) *' ' + "]"
        elif step > total_steps:
            progress_bar = "[" + progress_length*'=' + "=" + "]"
        else:
            progress_bar = "[" + i*'=' + '>' + (progress_length - i) *' ' + "]"
        # precentage display
        precentage = "{}".format(int(step * 100 / total_steps))
        precentage_len = len(precentage)
        precentage = (3-precentage_len)*" " + precentage
        # progress steps display
        total_steps_len = len(str(total_steps))
        steps_len = len(str(step))
        diff = total_steps_len - steps_len
        steps_str = diff*" " + str(step)
        progress_steps = steps_str + "/{}".format(total_steps)
        progress = " {}% {} {}  ".format(precentage, progress_bar, progress_steps)
    else:
        progress = ""
    data = ""
    for key, value in metrics.items():
        data += "" if value is None else key + ": {:.3f}  ".format(value)
    data = data[:-2]
    data += "" if time is None else "  time: {:.3f}".format(float(time))
    data += "" if lr is None else "  lr: {}".format(lr)

    statement = progress + data
    print(statement, end='\r', flush=False)