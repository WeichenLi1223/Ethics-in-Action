# Install
conda install pytorch torchvision torchaudio -c pytorch

Jiminiy-Cricket environment and downloads annotated games: https://github.com/hendrycks/jiminy-cricket

Sentencepiece: pip install sentencepiece

# RUN
python train.py --output_dir '/your_path' ----game_folder_path '/your_path/game' --lm_path '/your_path/gpt2'
