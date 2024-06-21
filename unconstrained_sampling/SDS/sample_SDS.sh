gpu=0

# python main.py exp.load_img_id=True\
#         +exp.img_path="/home/nzilberstein/code_submission/Inverse_submission/_exp/input/FFHQ" \
#         +exp.img_id="00010.png" \
#         +exp.gpu=$gpu

python main.py exp.load_text=False \
                +exp.csv_path="dataset/datacoco/subset.csv" \
                +exp.gpu=$gpu