import torch
import torchvision
from dataset import FHPsAOPMSBDataset
from torch.utils.data import DataLoader
DEVICE = "cuda"
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=False

):
    train_ds = FHPsAOPMSBDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_ds = FHPsAOPMSBDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

def get_loaders_test(
        test_dir,
        test_maskdir,
        test_transform
):
    test_ds = FHPsAOPMSBDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_ds,
    )
    return test_loader

def eval(loader, modelclientFE,modelserver,modelclientBE,loss_fn):
    #modelclientFE.eval()
    #modelserver.eval()
    #modelclientBE.eval()
    val_running_loss = 0.0
    valid_running_correct = 0.0
    valid_iou_score = 0.0
    valid_iou_score_class0 = 0.0
    valid_iou_score_class1 = 0.0
    valid_iou_score_class2 = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    with torch.no_grad():
        for idx,(x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.type(torch.LongTensor).to(device=DEVICE)
            enc1,predictions1 = modelclientFE(x)
            predictions2 = modelserver(predictions1)
            predictions3 = modelclientBE(enc1,predictions2)

            loss = loss_fn(predictions3, y)
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            valid_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            val_running_loss += loss.item()
            valid_iou_score += jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            valid_iou_score_class0 +=  iou_sklearn[0]
            valid_iou_score_class1 += iou_sklearn[1]
            valid_iou_score_class2 += iou_sklearn[2]
            #torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0,scale_each=True,normalize=True)

    #print(confusion_matrix)
    epoch_loss = val_running_loss / len(loader.dataset)
    epoch_acc = 100. * (valid_running_correct / len(loader.dataset))
    epoch_iou = (valid_iou_score / len(loader.dataset))
    epoch_iou_class0 = (valid_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (valid_iou_score_class1 / len(loader.dataset))
    epoch_iou_class2 = (valid_iou_score_class2 / len(loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1 +epoch_iou_class2 ) / 3
    epoch_iou_nobackground = (epoch_iou_class0 + epoch_iou_class1)/2
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1,epoch_iou_class2
    #model.train()


def test(loader, modelclientFE,modelserver,modelclientBE,loss_fn,folder):
    #modelclientFE.eval()
    #modelserver.eval()
    #modelclientBE.eval()
    test_running_loss = 0.0
    test_running_correct = 0.0
    test_iou_score = 0.0
    test_iou_score_class0 = 0.0
    test_iou_score_class1 = 0.0
    test_iou_score_class2 = 0.0
    test_accuracy = 0.0
    test_f1_score = 0.0
    allpredictions1 = []
    with torch.no_grad():
        for idx,(x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.type(torch.LongTensor).to(device=DEVICE)
            enc1,predictions1 = modelclientFE(x)
            allpredictions1.append(predictions1)
            predictions2 = modelserver(predictions1)
            predictions3 = modelclientBE(enc1,predictions2)
            loss = loss_fn(predictions3, y)
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            test_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()

            test_running_loss += loss.item()

            # iou score
            test_iou_score += jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            test_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            test_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            test_iou_score_class0 += iou_sklearn[0]
            test_iou_score_class1 += iou_sklearn[1]
            test_iou_score_class2 += iou_sklearn[2]
            #torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0,scale_each=True,normalize=True)

    epoch_loss = test_running_loss / len(loader.dataset)
    epoch_acc = 100. * (test_running_correct / len(loader.dataset))
    epoch_iou = (test_iou_score / len(loader.dataset))
    epoch_iou_class0 = (test_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (test_iou_score_class1 / len(loader.dataset))
    epoch_iou_class2 = (test_iou_score_class2 / len(loader.dataset))
    epoch_accuracy = (test_accuracy / len(loader.dataset))
    epoch_f1score = (test_f1_score / len(loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1 +epoch_iou_class2 ) / 3
    epoch_iou_nobackground = (epoch_iou_class0 + epoch_iou_class1)/2
    print("Testing accuracy score:", epoch_accuracy)
    print("Testing dice score/f1 Score:", epoch_f1score)
    print("IoU of Background:", epoch_iou_class0)
    print("IoU of tumor:", epoch_iou_class1)
    torch.save(allpredictions1, '/lustre06/project/6008975/csj5/narvalenv/allpredictions1_Fetus.pth')
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground
    #model.train()

