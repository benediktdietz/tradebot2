import sys, os
import numpy as np
import pandas as pd

from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from torch.utils.tensorboard import SummaryWriter

import chainer, chainerrl
from chainer import functions as F
from chainer import links as L
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainer import serializers

from env.StockTradingEnv import StockTradingEnv

import quandl
quandl.ApiConfig.api_key = "g-jhivSzUeDwajNsoqKd"


aluminum_shanghai = 'CHRIS/SHFE_AL'
gold_shanghai = 'CHRIS/SHFE_AU'
soybean_oil = 'CHRIS/CME_BO'
corn = 'CHRIS/CME_C'
canadian_dollar = 'CHRIS/CME_CD'
copper_shanghai = 'CHRIS/SHFE_CU'
dax_futures = 'CHRIS/EUREX_FDAX'
palladium = 'CHRIS/CME_PA'
ten_year_treasury = 'CHRIS/CME_TY'





class rl_stock_trader():

	def __init__(self):


		self.tb_outdir = './tensorboard_new/'
		self.tb_outdir_train = './tensorboard_new/train/'
		self.tb_outdir_test = './tensorboard_new/test/'
		self.model_outdir = self.tb_outdir

		self.training_counter = 0

		try: 
			sys.makedirs(self.tb_outdir_train)
			sys.makedirs(self.tb_outdir_test)
		except Exception: 
			pass


		self.writer_train = SummaryWriter(self.tb_outdir_train)
		self.writer_test = SummaryWriter(self.tb_outdir_test)


		self.monitor_freq = 50
		self.testing_samples = 1000

		self.validation_scores = []

		self.settings = {
			'past_horzion': 100,
			'inital_account_balance': 1e4,
			'stop_below_balance': 1e3,
			'transation_fee': .01,
			'years_training': 5,
			'years_testing': 1,
			}


		testing_end = date.today()
		testing_beginning = testing_end - relativedelta(years=self.settings['years_testing']) - relativedelta(days=self.settings['past_horzion'])
		training_end = testing_beginning
		training_beginning = training_end - relativedelta(years=self.settings['years_training']) - relativedelta(days=self.settings['past_horzion'])


		self.train_data = self.get_prices(gold_shanghai, 1, training_beginning, training_end)
		self.test_data = self.get_prices(gold_shanghai, 1, testing_beginning, testing_end)


		self.env_train = StockTradingEnv(self.train_data, self.settings, test=False)
		self.env_test = StockTradingEnv(self.test_data, self.settings, test=True)



	def get_prices(self, index, depth, start, end):

		data_prices = quandl.get(
			index + str(depth), 
			start_date = start, 
			end_date = end)

		data_prices.index = pd.to_datetime(data_prices.index)

		return data_prices


	def rl_agent(self, env):

		self.policy = chainer.Sequential(
			L.BatchNormalization(axis=0),
			L.Linear(None, 256),
			F.tanh,
			L.Linear(None, 128),
			F.tanh,
			# L.Linear(None, env.action_space.low.size, initialW=winit_last),
			L.Linear(None, env.action_space.low.size),
			# F.sigmoid,
			chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
				action_size=env.action_space.low.size,
				var_type='diagonal',
				var_func=lambda x: F.exp(2 * x),  # Parameterize log std
				# var_param_init=0,  # log std = 0 => std = 1
				))



		self.vf = chainer.Sequential(
			L.BatchNormalization(axis=0),
			L.Linear(None, 256),
			F.tanh,
			L.Linear(None, 128),
			F.tanh,
			L.Linear(None, 1),
		)

		# Combine a policy and a value function into a single model
		self.model = chainerrl.links.Branched(self.policy, self.vf)

		self.opt = chainer.optimizers.Adam(alpha=3e-4, eps=1e-5)
		self.opt.setup(self.model)


		self.agent = PPO(self.model, 
					self.opt,
					# obs_normalizer=obs_normalizer,
					gpu=-1,
					update_interval=512,
					minibatch_size=8,
					clip_eps_vf=None, 
					entropy_coef=0.001,
					# standardize_advantages=args.standardize_advantages,
					)

		return self.agent

	def monitor_training(self, tb_writer, t, i, done, action, monitor_data, counter):

		if t == 0 or i == 0:

			self.cash_dummy = []
			self.equity_dummy = []
			self.shares_dummy = []
			self.shares_value_dummy = []
			self.action_dummy = []
			self.action_prob_dummy = []

		self.cash_dummy.append(monitor_data['cash'])
		self.equity_dummy.append(monitor_data['equity'])
		self.shares_dummy.append(monitor_data['shares_held'])
		self.shares_value_dummy.append(monitor_data['value_in_shares'])
		self.action_dummy.append(monitor_data['action'])
		self.action_prob_dummy.append(monitor_data['action_prob'])

		if done:
			tb_writer.add_scalar('cash', np.mean(self.cash_dummy), counter)
			tb_writer.add_scalar('equity', np.mean(self.equity_dummy), counter)
			tb_writer.add_scalar('shares_held', np.mean(self.shares_dummy), counter)
			tb_writer.add_scalar('shares_value', np.mean(self.shares_value_dummy), counter)
			tb_writer.add_scalar('action', np.mean(self.action_dummy), counter)
			tb_writer.add_histogram('action_prob', np.mean(self.action_prob_dummy), counter)

	def plot_validation_figures(self, index, name):

		area_plots = []
		box_data = []
		for j in range(len(np.unique(np.asarray(self.validation_scores)[:,0]))):
			dummy = np.asarray(self.validation_scores)[:,index][np.where(np.asarray(self.validation_scores)[:,0] == np.unique(np.asarray(self.validation_scores)[:,0])[j])]
			box_data.append(dummy)
			area_plots.append(
				[
				np.percentile(dummy, 5),
				np.percentile(dummy, 25),
				np.percentile(dummy, 50),
				np.percentile(dummy, 75),
				np.percentile(dummy, 95),
				])
		area_plots = np.asarray(area_plots)
		
		
		p05 = area_plots[:,0]
		p25 = area_plots[:,1]
		p50 = area_plots[:,2]
		p75 = area_plots[:,3]
		p95 = area_plots[:,4]

		plt.figure(figsize=(18,18))
		plt.fill_between(np.arange(area_plots.shape[0]), p05, p95, facecolor='darkgreen', alpha=.3)
		plt.fill_between(np.arange(area_plots.shape[0]), p25, p75, facecolor='darkgreen', alpha=.8)
		plt.plot(p50, linewidth=3, color='lightblue')
		plt.grid()
		plt.title(name + ' equity statistics over 1 year')
		plt.xlabel('trained episodes')
		plt.ylabel('equity [$]')
		plt.savefig(self.tb_outdir + 'area_' + name + '_equity.pdf')
		plt.close()

		plt.figure(figsize=(18,18))
		plt.boxplot(
			box_data, 
			notch=None, 
			sym=None,
			vert=None, 
			whis=None, 
			positions=None, 
			widths=None, 
			patch_artist=None, 
			bootstrap=None, 
			usermedians=None, 
			conf_intervals=None, 
			meanline=None, 
			showmeans=None, 
			showcaps=None, 
			showbox=None, 
			showfliers=None, 
			boxprops=None, 
			labels=None, 
			flierprops=None, 
			medianprops=None, 
			meanprops=None, 
			capprops=None, 
			whiskerprops=None, 
			manage_ticks=True, 
			autorange=False, 
			zorder=None)
		plt.grid()
		plt.title('equity statistics over 1 year')
		plt.xlabel('trained episodes')
		plt.ylabel('equity [$]')
		plt.savefig(self.tb_outdir + 'box_' + name + '_equity.pdf')
		plt.close()

	def validate(self, agent, episode, counter):

		test_equity = []
		test_equity_i = []

		plt.figure(figsize=(18,18))

		for i in range(0, self.testing_samples):

			obs = self.env_test.reset()

			reward = 0
			done = False
			R = 0  # return (sum of rewards)
			t = 0  # time step


			while not done:
				action = agent.act(obs)
				obs, reward, done, _, monitor_data = self.env_test.step(action)
				test_equity.append(monitor_data['equity'])
				self.monitor_training(self.writer_test, t, i, done, action, monitor_data, counter)

				R += reward
				t += 1

				if done:
					test_equity = test_equity[:-1]
					plt.plot(test_equity[:-1], linewidth=1)
					self.validation_scores.append([counter, np.mean(test_equity), np.amin(test_equity), np.amax(test_equity), test_equity[-1]])
					test_equity = []

		benchmark = self.test_data['Close'].values[self.settings['past_horzion']:]
		benchmark /= benchmark[0]
		benchmark *= self.settings['inital_account_balance']

		time_axis = self.test_data.index[self.settings['past_horzion']:].date
		time_axis_short = time_axis[::10]

		plt.plot(benchmark , linewidth=3, color='k', label='close')
		plt.xticks(np.linspace(0, len(time_axis), len(time_axis_short)-1), time_axis_short, rotation=90)
		plt.grid()
		plt.title('validation runs at episode' + str(episode))
		plt.xlabel('episode')
		plt.ylabel('equity [$]')
		plt.legend()
		plt.savefig(self.tb_outdir + 'validation_' + str(episode) + '.pdf')
		plt.close()

		self.plot_validation_figures(1, 'mean')
		self.plot_validation_figures(2, 'min')
		self.plot_validation_figures(3, 'max')
		self.plot_validation_figures(4, 'final')



	def train(self):


		print('\nstart training loop\n')

		def check_types(input, inputname):
			if np.isnan(input).any(): print('----> ', inputname, ' array contains NaN\n', np.isnan(input).shape, '\n')
			if np.isinf(input).any(): print('----> ', inputname, ' array contains inf\n', np.isinf(input).shape, '\n')



		self.agent = self.rl_agent(self.env_train)


		n_episodes = int(1e5)

		log_data = []


		for i in range(0, n_episodes + 1):


			obs = self.env_train.reset()

			reward = 0
			done = False
			R = 0  # return (sum of rewards)
			t = 0  # time step


			while not done:

				# self.env.render()
				action = self.agent.act_and_train(obs, reward)

				obs, reward, done, _, monitor_data = self.env_train.step(action)

				self.monitor_training(self.writer_train, t, i, done, action, monitor_data, self.training_counter)

				R += reward
				t += 1


				if t % 10 == 0 and not done:
					log_data.append({
						'equity': int(monitor_data['equity']),
						'shares_held': int(monitor_data['shares_held']),
						'shares_value': int(monitor_data['value_in_shares']),
						'cash': int(monitor_data['cash']),
						't': int(t),
						})
			
			
				if done:
					if i % 10 == 0:
						print('\nrollout ' + str(i) + '\n', pd.DataFrame(log_data).mean())
					log_data = []
					self.training_counter += 1

			if i % self.monitor_freq == 0:

				# print('\n\nvalidation...')
				self.validate(self.agent, i, self.training_counter)


			if i % 100 == 0 and i > 0:

				self.agent.save(self.model_outdir)

				serializers.save_npz(self.model_outdir + 'model.npz', self.model)

			
			# if i % 1000 == 0:
			#     print('\nepisode:', i, ' | episode length: ', t, '\nreward:', R,
			#           '\nstatistics:', self.agent.get_statistics(), '\n')

		self.agent.stop_episode_and_train(obs, reward, done)
		print('Finished.')


rl_stock_trader().train()


