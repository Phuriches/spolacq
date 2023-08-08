#%% ------------------------------------ import ------------------------------------
import numpy as np
import torch
import yaml
import json
import resampy
import os
import sys

from glob import glob

# from stable_baselines3 import TD3
# from stable_baselines3.common.buffers import ReplayBuffer
# from stable_baselines3.common.noise import ActionNoise
# from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
# from stable_baselines3.common.policies import BaseModel, BasePolicy
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, get_actor_critic_arch
# from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# from stable_baselines3.td3.policies import TD3Policy

from asr_api import ASR
from sb3_api import CustomDDPG
from gym_api import SpoLacq3

sys.path.append("../tools/I2U")
from models_i2u import ImageToUnit

sys.path.append("../tools/U2S")
from hparams import create_hparams
from text import text_to_sequence
from train import load_model

sys.path.append("../tools/hifi-gan")
from env import AttrDict
from inference import load_checkpoint
from models_hifi_gan import Generator

"""
An agent learn to understand and speak under an environment.
How to evaluate it?
- accuracy: the percentage of correct sentences
- fluency: the percentage of fluent sentences
- diversity: the percentage of diverse sentences
- meaning: the percentage of sentences with the same meaning
- style: the way of speaking
- voice: the voice of speaker
- emotion: the emotion of speaker
- accent: the accent of speaker
- language: the language of speaker
- rate of reward: the rate of reward
    - If agent can maximize the reward to 1, it can understand and speak.
    - If between, some sentences are correct, some are incorrect.
    - If agent can maximize the reward to 0, it cannot understand and speak.

TODO:
- Logging:
    - Image path, internal state, action (speech and transcription), and reward.
- How to test the trained agent?
    - Test it on the test set.
    - Extract those information (like image, state, action, and reward)
"""
# load the agent's brain

class Args:
    def __init__(self, word_map, u2s_config):
        self.word_map = word_map
        self.u2s_config = u2s_config

def make_action2text(config, args):
    word_map = args.word_map # custom word map
    u2s_config = args.u2s_config # custom u2s config
    assert word_map in ['default', 'word_map_20', 'word_map_100',
                        'word_map_5_hubert', 'word_map_10_hubert',
                        'word_map_20_hubert', 'word_map_100_hubert',
                        'word_map_spokencoco', 'word_map_rawfood_100']

    model_path = {
        'word_map'             : 'model_path',
        'word_map_20'          : 'model_path_20',
        'word_map_100'         : 'model_path_100',
        'word_map_5_hubert'    : 'model_path_5_hubert',
        'word_map_10_hubert'   : 'model_path_10_hubert',
        'word_map_20_hubert'   : 'model_path_20_hubert',
        'word_map_100_hubert'  : 'model_path_100_hubert',
        'word_map_rawfood_100' : 'model_path_rawfood_100',
    }[word_map]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # I2U
    word_map_path = os.path.join(os.path.dirname(__file__), "..", config["I2U"][word_map])
    image_to_unit_model_path = os.path.join(os.path.dirname(__file__), "..", config["I2U"][model_path])
    print(f'----------- model path -----------')
    print(f'model config path: {model_path}')
    print(f'model path: {image_to_unit_model_path}')
    print(f'----------------------------------')
    with open(word_map_path) as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    special_words = {"<unk>", "<start>", "<end>", "<pad>"}
    print(f'length word map: {len(word_map)}')

    image_to_unit = ImageToUnit(word_map, max_len=102)
    image_to_unit.load_state_dict(torch.load(image_to_unit_model_path))
    image_to_unit.to(device).eval()

    # U2S
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", config["U2S"][u2s_config])
    print(f'checkpoint_path: {checkpoint_path}')
    hparams = create_hparams()
    hparams.n_symbols = len(word_map) - 2
    # hparams.n_symbols = 152
    print(f'length hparams.n_symbols: {hparams.n_symbols}')
    tacotron2 = load_model(hparams)
    tacotron2.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    tacotron2.to(device).eval()

    # hifi-gan
    hifi_gan_config_path = os.path.join(os.path.dirname(__file__), "..", config["HiFi_GAN"]["config"])
    hifi_gan_checkpoint_path = os.path.join(os.path.dirname(__file__), "..", config["HiFi_GAN"]["checkpoint"])

    with open(hifi_gan_config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(hifi_gan_checkpoint_path, device)
    generator.load_state_dict(state_dict_g["generator"])

    # ASR
    asr_dir = os.path.join(os.path.dirname(__file__), "..", config["ASR"]["dir"])
    asr = ASR(asr_dir)

    @torch.inference_mode()
    def action2text(action):
        action = torch.from_numpy(action).unsqueeze(0).to(device)

        unit_seq = image_to_unit.infer(action=action, beam_size=config["I2U"]["beam_size"])
        words = [rev_word_map[idx] for idx in unit_seq if rev_word_map[idx] not in special_words]

        sequence = np.array(text_to_sequence(" ".join(words), ["english_cleaners"]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()

        try:
            _, mel_outputs_postnet, _, _ = tacotron2.inference(sequence)

            audio = generator(mel_outputs_postnet)
            audio = audio.squeeze().cpu().numpy().astype(np.float64)
            audio = resampy.resample(audio, 22050, 16000)

            transcript = asr(audio)
            print(f'transcript: {transcript}', flush=True)

        except RuntimeError as e:
            transcript = ""
            print(e, flush=True)

        return transcript

    return action2text

def test(num_episode: int, env, model: CustomDDPG) -> None:
    """Test the learnt agent."""

    for i in range(num_episode):
        print(f"episode {i}", "-" * 40)
        state = env.reset()
        print(f'state: {state["state"]}')
        print(f'state keys: {state.keys()}')
        total_reward = 0
        while True:
            print(f"-------------- state: {state.keys} --------------")
            # render the state
            # env.render()

            # Agent gets an environment state and returns a decided action
            action, _ = model.predict(state, deterministic=True)
            print(f'action shape: {action.shape}')
            print('action', action)
            # Environment gets an action from the agent, proceeds the time step,
            # and returns the new state and reward etc.
            state, reward, done, info = env.step(action)
            total_reward += reward
            # utterance = env.dlgworld.asr(env.sounddic[action])
            # print(f"utterance: {utterance}, reward: {reward}")

            if done:
                print(f"total_reward: {total_reward}\n")
                break


#%% load environment
# init args
args = Args(word_map="word_map_100_hubert", u2s_config="u2s_path_haoyuan")

with open("../conf/spolacq3.yaml") as y:
    config = yaml.safe_load(y)

action2text = make_action2text(config, args)
# load pretrained agent
model_path = "../../models/RL_hubert/23-06-25_00-09-40_pikaia30_word_map_100_hubert/"
model = CustomDDPG.load(f"{model_path}best_model")
eval_env = SpoLacq3(
        glob("../data/dataset/*/test_number[12]/*.jpg"),
        d_image_features=768,
        d_embed=config["I2U"]["d_embed"],
        action2text=action2text,
    )

#%% test state
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
state = eval_env.reset()
state['leftimage'], state['rightimage']

img = np.reshape(state['leftimage'], (224, 224 ,3))
img.shape
def norm(img):
    return (255*(img-img.min())/(img.max()-img.min())).astype(np.int16)

left_img = np.reshape(state['leftimage'], (224, 224 ,3))
right_img = np.reshape(state['rightimage'], (224, 224 ,3))
left_img = norm(left_img)
right_img = norm(right_img)

plt.subplot(1, 2, 1)
plt.imshow(left_img)
plt.title('left image')
plt.subplot(1, 2, 2)
plt.imshow(right_img)
plt.title('right image')
#%% what is the internal state?
state['state']
# why motivation of the agnet is the mean of the image?


# %% ------------------------------------ test ------------------------------------
# we wanna log the image, internal state, action (speech and transcription), and reward.
# get correct transcription

image_list = glob("../data/dataset/*/test_number[12]/*.jpg")
sample_img = image_list[0] # one banana
print(sample_img.split('/')[3].upper())

plt.imshow(Image.open(sample_img))
plt.show()


# %% ------------------------------------ run environment ------------------------------------
test(
    num_episode=10,
    env=eval_env,
    model=model,
)
# ---------------------------------------------------------------------------------------------

# %%
