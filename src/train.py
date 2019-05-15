from torch.optim.lr_scheduler import StepLR
import logging
import nerual_net
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from loader import get_loader
import lfw_verification
import common_args

logging.basicConfig(filename='FYP_Resnet.log', level=logging.INFO)
hr_model_root = '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/30000_net_backbone_asoftmax_64.pth'
lr_model_root = '/home/tangjiawei/project/fyp/saved/30000_net_backbone.pth'
if __name__ == '__main__':
    args = common_args.get_args()

    # setup device and dataloader
    device = torch.device(1)

    print(device)
    dataloader = get_loader('webface', batch_size=128, N=args.downsampling_factor)

    # Setup network Spherenet10_HR
    hr_spherenet = nerual_net.spherenet(64, args.feature_dim, args.use_pool, args.use_dropout)
    hr_spherenet.load_state_dict(torch.load(hr_model_root))
    hr_spherenet.to(device)

    # # Setup network Spherenet20_HR
    # num_layers = 20
    # hr_spherenet = nerual_net.spherenet(num_layers, args.feature_dim, args.use_pool, args.use_dropout)
    # hr_spherenet.load_state_dict(torch.load(model_root))
    # hr_spherenet.to(device)

    # Setup network Spherenet20_lR
    lr_spherenet = nerual_net.spherenet(20, args.feature_dim, args.use_pool, args.use_dropout)
    lr_spherenet.load_state_dict(torch.load(lr_model_root))
    lr_spherenet.to(device)

    # #setup network sphereface64_HR
    # hr_spherenet_64 = nerual_net.spherenet(64, args.feature_dim, args.use_pool, args.use_dropout)
    # hr_spherenet_64.load_state_dict(
    #     torch.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/30000_net_backbone_asoftmax_64.pth'))
    # hr_spherenet_64.to(device)

    # # setup network sphereface64_LR
    # lr_spherenet_64 = nerual_net.spherenet(64, args.feature_dim, args.use_pool, args.use_dropout)
    # lr_spherenet_64.load_state_dict(
    #     torch.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/30000_net_backbone_asoftmax_64.pth'))
    # lr_spherenet_64.to(device)

    # setup criterion and optimizer
    criterion_lr = torch.nn.MSELoss()
    optimizer = optim.Adam(lr_spherenet.parameters(),
                           lr=0.0003,
                           weight_decay=0.0005,
                           betas=(0.9, 0.999),
                           amsgrad=True)
    scheduler = StepLR(optimizer, step_size=20001, gamma=0.1)



    # ------------------------
    # Start Training and Validating
    # ------------------------
    logging.info("# ------------------------")
    logging.info("Start training LR_Spherentface 64 with {}x{} probe image".format(int(96 / args.downsampling_factor),
                                                                int(112 / args.downsampling_factor)))
    logging.info("# ------------------------")
    lfw_verification.run(lr_spherenet, feature_dim=512, device=device, N=args.downsampling_factor)
    pbar = tqdm(range(1, args.iterations + 1), ncols=0)
    for total_steps in pbar:
        lr_spherenet.train()
        hr_spherenet.eval()
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        hr_face, lr_face, target = dataloader.next()
        hr_features = hr_spherenet(hr_face.to(device))
        lr_features = lr_spherenet(lr_face.to(device))
        target = target.to(device)
        loss = criterion_lr(hr_features, lr_features)
        loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        description = '{}: {:.4f} , {}: {:.4f}'.format('loss', loss, 'lr', lr)
        logging.info(description)
        pbar.set_description(desc=description)
        if total_steps == 20 or total_steps == 200 or total_steps == 300 or total_steps == 400 or total_steps == 500 or total_steps == 600 or total_steps == 700 or total_steps == 800 or total_steps == 900 or total_steps == 1000 or total_steps == 1200 or total_steps == 1600 or total_steps == 2000 or total_steps == 2400 or total_steps == 2800 or total_steps == args.iterations:
            torch.save(lr_spherenet.state_dict(),
                       '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_spherenet_hr64_lr20_model/{}_lr_spherenet2064_df{}.pth'.format(
                           total_steps, args.downsampling_factor))
            print("HR image result of LR spherenet")
            logging.info("HR image result of LR spherenet")
            lfw_verification.run(lr_spherenet, feature_dim=512, device=device, N=1)
            print("LR image result of LR spherenet")
            logging.info("LR image result of LR spherenet")
            lfw_verification.run(lr_spherenet, feature_dim=512, device=device, N=9)
