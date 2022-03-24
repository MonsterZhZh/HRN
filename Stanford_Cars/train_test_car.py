import torch
from utils_car import *
import copy
import time
from sklearn.metrics import confusion_matrix, average_precision_score



def train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, CELoss, tree, device, devices, save_name, dataset):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)
    for epoch in range(epoches):
        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        order_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        species_total= 0

        idx = 0
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, target_list_sig = get_order_family_target(targets, dataset, device)

            optimizer.zero_grad()

            if len(devices) > 1:
                xc1_sig, xc3, xc3_sig = netp(inputs)
            else:
                xc1_sig, xc3, xc3_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'Car':
                leaf_labels = torch.nonzero(targets > 8, as_tuple=False)

            if leaf_labels.shape[0] > 0:
                if dataset == 'Car':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 9
                select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss
                
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
            with torch.no_grad():
                _, order_predicted = torch.max(xc1_sig.data, 1)
                order_total += order_targets.size(0)
                order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

                if leaf_labels.shape[0] > 0:
                    select_xc3 = torch.index_select(xc3, 0, leaf_labels.squeeze())
                    select_xc3_sig = torch.index_select(xc3_sig, 0, leaf_labels.squeeze())
                    _, species_predicted_soft = torch.max(select_xc3.data, 1)
                    _, species_predicted_sig = torch.max(select_xc3_sig.data, 1)
                    species_total += select_leaf_labels.size(0)
                    species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                    species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()

        if lr_adjt == 'Step':
            scheduler.step()

        train_order_acc = 100.*order_correct/order_total
        train_species_acc_soft = 100.*species_correct_soft/species_total
        train_species_acc_sig = 100.*species_correct_sig/species_total
        train_loss = train_loss/(idx+1)
        epoch_end = time.time()
        print('Iteration %d, train_order_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
            (epoch, train_order_acc, train_species_acc_soft, train_species_acc_sig, train_loss, (epoch_end - epoch_start)))

        test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss = test(net, testloader, CELoss, tree, device, dataset)
        
        if test_species_acc_soft > max_val_acc:
            max_val_acc = test_species_acc_soft
            best_epoch = epoch
            net.cpu()
            torch.save(net, './Cars/model_'+save_name+'.pt')
            net.to(device)

    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))


def test(net, testloader, CELoss, tree, device, dataset):
    epoch_start = time.time()
    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        species_total= 0

        idx = 0
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, target_list_sig = get_order_family_target(targets, dataset, device)

            xc1_sig, xc3, xc3_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'Car':
                leaf_labels = torch.nonzero(targets > 8, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 9
            
            select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
            ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
            loss = ce_loss_species + tree_loss

            test_loss += loss.item()
    
            _, order_predicted = torch.max(xc1_sig.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, species_predicted_soft = torch.max(xc3.data, 1)
            _, species_predicted_sig = torch.max(xc3_sig.data, 1)
            species_total += select_leaf_labels.size(0)
            species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
            species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()


        test_order_acc = 100.* order_correct/order_total
        test_species_acc_soft = 100.* species_correct_soft/species_total
        test_species_acc_sig = 100.* species_correct_sig/species_total
        test_loss = test_loss/(idx+1)
        epoch_end = time.time()
        print('test_order_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_loss = %.6f, Time = %.4s' % \
             (test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss, epoch_end - epoch_start))

    return test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss



def test_AP(model, dataset, test_set, test_data_loader, device):
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            select_labels = labels[:, test_set.to_eval]
            if dataset == 'CUB' or dataset == 'Air':
                y_order_sig, y_family_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, y_family_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data
            else:
                y_order_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data
            
            predicted = batch_pMargin > 0.5
            total += select_labels.size(0) * select_labels.size(1)
            correct += (predicted.to(torch.float64) == select_labels).sum()
            cpu_batch_pMargin = batch_pMargin.to('cpu')
            y = select_labels.to('cpu')
            if j == 0:
                test = cpu_batch_pMargin
                test_y = y
            else:
                test = torch.cat((test, cpu_batch_pMargin), dim=0)
                test_y = torch.cat((test_y, y), dim=0)
        score = average_precision_score(test_y, test, average='micro')
        print('Accuracy:' + str(float(correct) / float(total)))
        print('Precision score:' + str(score))