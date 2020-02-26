from modelalgo import *

# Settings
input_dim = 72
output_dim = 2
nseqlen = 256

test_dir =  "LMBTrain/untitled folder/untitled folder/csv/test/"
train_dir = "LMBTrain/untitled folder/untitled folder/csv/train/"

# Load data
inputs, outputs, ids = load_data(train_dir, input_dim, output_dim, nseqlen, nsamples = 900)
inputs_test, outputs_test, ids_test = load_data(test_dir, input_dim, output_dim, nseqlen, nsamples = 900)

ntrain = inputs.shape[0]

inputs = np.concatenate((inputs, inputs_test), axis=0)
outputs = np.concatenate((outputs, outputs_test), axis=0)
ids = np.concatenate((ids, ids_test), axis=0)

n = inputs.shape[0] + inputs_test.shape[0]

# Combinations
hiddens = [128]
lstmlays = [3]
products = list(itertools.product(hiddens, lstmlays))

# Run
results = {}
tt = time.strftime('%Y%m%d%H%M%s')

for exam in products:

    # needs to try a few combinations
    # cols = range(72) # everything
    cols = range(12) + [21 + i for i in range(6)] + [28 + i for i in range(6)] # Traj + Vel(
    model = construct_model(hidden=48, lstm_layers=3, input_dim=len(cols), output_dim=1)
    history = model.fit(inputs[0:ntrain, :, cols], outputs[0:ntrain, :, 1:2], nb_epoch=500, batch_size=32, verbose=2,validation_split=0.2)
    plot_history(history)

    # Test on the test set
    scores = model.evaluate(inputs[ntrain:n, :, cols], outputs[ntrain:n, :, 1:2])
    print("Test set loss: %.4f" % (scores[0]))
    sdist = []
    res = model.predict(inputs[ntrain:n, :, cols])

    for ntrial in range(7):
        likelihood = res[ntrial, :, 0:1]
        true = outputs[ntrain + ntrial, :, 1:2]
        d = eval_prediction(likelihood, true, ids[ntrain + ntrial], plot=False, shift=0)
        sdist.extend(d)
        if (d[0] > 10):
            eval_prediction(likelihood, true, ids[ntrain + ntrial], plot=True, shift=0)

    plot_stats(sdist)

    # matplotlib inline

    sdist = []
    files = os.listdir(test_dir)

    for i, filename in enumerate(files):
        R = np.loadtxt("%s/%s" % (test_dir, filename), delimiter=',')
        X = R[:, cols]
        true = R[:, (input_dim + 1):(input_dim + 2)]
        likelihood = model.predict(X.reshape((1, -1, len(cols))))[0]

        sdist.extend(eval_prediction(likelihood, true, filename))

        if i > 20:
            break

    results[exam] = (history.history, scores[0])
    model.save("model-%s-%d-%d.h5" % (tt, exam[0], exam[1]))

pickle.dump(results, open("res-%s.p" % (tt,), "wb"))
