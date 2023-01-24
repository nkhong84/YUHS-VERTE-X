
def test_score(batch_te, device, model, args):
    model.eval()

    res_dict = {}
    pred_labels = []
    pid_labels = []

    for j, (x_te,c_te,label) in enumerate(batch_te):
        x_te = x_te.to(device)
        c_te = c_te.to(device).float()

        pred_te = model(x_te,c_te)

        pid_labels.extend(label)
        pred_labels.extend(pred_te.cpu().detach().numpy())

    for pid,yp in zip(pid_labels,pred_labels):
        res_dict[pid] = yp[0]

    return res_dict

