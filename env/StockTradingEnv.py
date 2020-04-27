import random
import json
import gym
import sys
from gym import spaces
import pandas as pd
import numpy as np
from scipy.special import softmax


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, settings, test):
        super(StockTradingEnv, self).__init__()

        self.settings = settings
        self.test = test

        self.df = df

        self.past_horzion = int(self.settings['past_horzion'])

        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min() + 1e-4)


        self.df.fillna(method ='pad') 
        self.df.fillna(0.) 
        self.df.replace(to_replace = np.nan, value = 0.)

        self.first_day = self.df.first_valid_index()


        self.action_space = spaces.Box(
            low = np.zeros(3), 
            high = np.ones(3), 
            dtype = np.float16)

        # self.observation_space = spaces.Box(
        #     np.zeros(self.df.shape[1]-1), 
        #     np.ones(self.df.shape[1]-1), 
        #     dtype=np.float16)

        self.observation_space = spaces.Box(
            np.zeros(self.past_horzion), 
            np.ones(self.past_horzion), 
            dtype=np.float16)


    def _next_observation(self):
    
        return self.df['Close'].iloc[self.current_step-self.past_horzion:self.current_step].fillna(0.).astype(np.float32)


    def _take_action(self, action):
     
        self.current_price_original = self.df['Close'].iloc[self.current_step]
        
        transation_fees = self.current_price_original * self.settings['transation_fee'] + .001

        self.action_probs = softmax(action)

        self.action = np.argmax(self.action_probs)

        # 0-buy | 1-hold | 2-sell
        if self.action == 0:

            num_possible = np.floor((self.cash - self.settings['stop_below_balance']) / (self.current_price_original + transation_fees))

            if num_possible >= 1:

                position_size = np.floor(self.action_probs[0] * num_possible)

                self.cash -= position_size * (self.current_price_original + transation_fees)
                self.shares_held += position_size

        if self.action == 2:

            if self.shares_held >= 1:

                position_size = np.int(np.round(self.action_probs[2] * self.shares_held))

                if position_size > 0:

                    self.cash += position_size * (self.current_price_original - transation_fees)
                    self.shares_held -= position_size

        self.equity = self.cash + self.shares_held * self.current_price_original
        self.value_in_shares = self.shares_held * self.current_price_original



    def step(self, action):

        self._take_action(action)

        self.current_step += 1       
        self.steps_taken += 1     

        reward = self.equity

        done = self.equity <= self.settings['stop_below_balance'] or self.current_step > len(self.df) - 1 or self.steps_taken > 365

        if done: self.reset()

        obs = self._next_observation()

        monitor_data = {
                'equity': self.equity,
                'shares_held': self.shares_held,
                'value_in_shares': self.value_in_shares,
                'cash': self.cash,
                'action': softmax(action),
                'action_prob': np.argmax(softmax(action)),
                }

        return obs, reward, done, {}, monitor_data


    def reset(self):

        self.cash = self.settings['inital_account_balance']
        self.net_worth = self.settings['inital_account_balance']

        self.shares_held = 0
        self.value_in_shares = 0

        self.steps_taken = 0

        # self.equity = []
        # self.equity.append(self.settings['inital_account_balance'])
        self.equity = self.settings['inital_account_balance']

        self.transations = 0
        
        if self.test: self.current_step = self.past_horzion
        else: self.current_step = random.randint(self.past_horzion, len(self.df) - self.past_horzion)

        return self._next_observation()


    def render(self, mode='human', close=False):

        profit = self.net_worth - self.settings['inital_account_balance']

        if self.current_step == 400:
      
            print('Step: ', int(self.current_step), 
                # '\nAction--->  type/amount/symbol ', self.action_type, '/ ', self.amount, '/ ', symbol,
                '\nBalance: ', np.round(self.balance, 1), 
                '\nShares held/purch./sold: ', int(np.sum(self.shares_held)), '/ ', int(np.sum(self.total_shares_purchased)), '/ ', int(np.sum(self.total_shares_sold)), 
                '\nFees: ', np.round(self.total_fees, 1),
                '\nProfit: ', np.round(profit, 1),
                end = '\n\r\n')








