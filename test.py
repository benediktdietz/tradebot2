import sys, os
import numpy as np
import pandas as pd

from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

import chainer, chainerrl
from chainer import functions as F
from chainer import links as L
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainer import serializers

os.system('clear')

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

		run_name = 'run_test'
		self.outdir = './results/' + run_name + '/'
		self.outdir_train = self.outdir + 'train/'
		self.outdir_test = self.outdir + 'test/'

		self.training_counter = 0

		try: 
			sys.makedirs(self.outdir_train)
			sys.makedirs(self.outdir_test)
		except Exception: 
			pass

		self.writer_train = SummaryWriter(self.outdir_train)
		self.writer_test = SummaryWriter(self.outdir_test)

		self.monitor_freq = 100
		self.testing_samples = 100

		self.validation_scores = []
		self.training_scores = []

		self.settings = {
			'past_horzion': 100,
			'max_steps': 365,
			'inital_account_balance': 1e4,
			'stop_below_balance': 1e3,
			'transation_fee': .1,
			'years_training': 5,
			'years_testing': 1,
			}

		testing_end = date.today()
		testing_beginning = testing_end - relativedelta(years=self.settings['years_testing']) - relativedelta(days=self.settings['past_horzion'])
		training_end = testing_beginning - relativedelta(days=1)
		training_beginning = training_end - relativedelta(years=self.settings['years_training']) - relativedelta(days=self.settings['past_horzion'])

		self.data = {
			'train_gold': self.get_prices(gold_shanghai, 1, training_beginning, training_end),
			'train_copper': self.get_prices(copper_shanghai, 1, training_beginning, training_end),
			'train_aluminum': self.get_prices(aluminum_shanghai, 1, training_beginning, training_end),
			'test_gold': self.get_prices(gold_shanghai, 1, testing_beginning, testing_end),
			'test_copper': self.get_prices(copper_shanghai, 1, testing_beginning, testing_end),
			'test_aluminum': self.get_prices(aluminum_shanghai, 1, testing_beginning, testing_end),
			'test_soybean_oil': self.get_prices(soybean_oil, 1, testing_beginning, testing_end),
			'test_dax_futures': self.get_prices(dax_futures, 1, testing_beginning, testing_end),
			'test_corn': self.get_prices(corn, 1, testing_beginning, testing_end),
			'test_canadian_dollar': self.get_prices(canadian_dollar, 1, testing_beginning, testing_end),
			}

		# print('\n\n*************\n', self.data['test_corn'], '\n\n')

		self.env_test_gold = StockTradingEnv(
			self.get_prices(gold_shanghai, 1, testing_beginning, testing_end), 
			self.settings, 
			test=True)
		self.env_test_copper = StockTradingEnv(
			self.get_prices(copper_shanghai, 1, testing_beginning, testing_end), 
			self.settings, 
			test=True)
		self.env_test_aluminum = StockTradingEnv(
			self.get_prices(aluminum_shanghai, 1, testing_beginning, testing_end), 
			self.settings, 
			test=True)
		self.env_test_soy_bean = StockTradingEnv(
			self.get_prices(soybean_oil, 1, testing_beginning, testing_end), 
			self.settings, 
			test=True)
		self.env_test_dax = StockTradingEnv(
			self.get_prices(dax_futures, 1, testing_beginning, testing_end), 
			self.settings, 
			test=True)
		self.env_test_corn = StockTradingEnv(
			self.get_prices(corn, 1, testing_beginning, testing_end), 
			self.settings, 
			test=True)
		self.env_test_canadian_dollar = StockTradingEnv(
			self.get_prices(canadian_dollar, 1, testing_beginning, testing_end), 
			self.settings, 
			test=True)

		self.env_train = StockTradingEnv(self.data['train_gold'], self.settings, test=False)
		# self.env_test = StockTradingEnv(self.data['test_gold'], self.settings, test=True)

		self.test_envs = {
			'gold': StockTradingEnv(self.data['test_gold'], self.settings, test=True),
			'copper': StockTradingEnv(self.data['test_copper'], self.settings, test=True),
			'aluminum': StockTradingEnv(self.data['test_aluminum'], self.settings, test=True),
			}


		self.agent = self.rl_agent(self.env_train)

	def get_prices(self, index, depth, start, end):

		data_prices = quandl.get(
			index + str(depth), 
			start_date = start, 
			end_date = end)

		data_prices.index = pd.to_datetime(data_prices.index)

		return data_prices

	def rl_agent(self, env):

		# self.policy = chainer.Sequential(
		# 	L.BatchNormalization(axis=0),
		# 	L.Linear(None, 256),
		# 	# F.dropout(ratio=.5),
		# 	F.tanh,
		# 	L.Linear(None, 128),
		# 	# F.dropout(ratio=.5),
		# 	F.tanh,
		# 	# L.Linear(None, env.action_space.low.size, initialW=winit_last),
		# 	L.Linear(None, env.action_space.low.size),
		# 	# F.sigmoid,
		# 	chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
		# 		action_size=env.action_space.low.size,
		# 		var_type='diagonal',
		# 		var_func=lambda x: F.exp(2 * x),  # Parameterize log std
		# 		# var_param_init=0,  # log std = 0 => std = 1
		# 		))

		self.policy = chainer.Sequential(
			L.BatchNormalization(axis=0),
			L.Linear(None, 256),
			# F.dropout(ratio=.5),
			F.sigmoid,
			# F.relu,
			L.Linear(None, 128),
			# F.dropout(ratio=.5),
			F.sigmoid,
			# L.Linear(None, env.action_space.low.size, initialW=winit_last),
			L.Linear(None, env.action_space.low.size),
			F.sigmoid,
			chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
				action_size=env.action_space.low.size,
				var_type='diagonal',
				var_func=lambda x: F.exp(2 * x),  # Parameterize log std
				# var_param_init=0,  # log std = 0 => std = 1
				))

		self.vf = chainer.Sequential(
			L.BatchNormalization(axis=0),
			L.Linear(None, 256),
			# F.dropout(ratio=.5),
			F.sigmoid,
			L.Linear(None, 128),
			# F.dropout(ratio=.5),
			F.sigmoid,
			L.Linear(None, 1),
			F.sigmoid,
		)

		# self.vf = chainer.Sequential(
		# 	L.BatchNormalization(axis=0),
		# 	L.Linear(None, 256),
		# 	# F.dropout(ratio=.5),
		# 	F.tanh,
		# 	L.Linear(None, 128),
		# 	# F.dropout(ratio=.5),
		# 	F.tanh,
		# 	L.Linear(None, 1),
		# )

		# Combine a policy and a value function into a single model
		self.model = chainerrl.links.Branched(self.policy, self.vf)

		self.opt = chainer.optimizers.Adam(alpha=3e-3, eps=1e-5)
		self.opt.setup(self.model)


		self.agent = PPO(self.model, 
					self.opt,
					# obs_normalizer=obs_normalizer,
					gpu=-1,
					update_interval=64,
					minibatch_size=32,
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

		# if done:
			# tb_writer.add_scalar('cash', np.mean(self.cash_dummy), counter)
			# tb_writer.add_scalar('equity', np.mean(self.equity_dummy), counter)
			# tb_writer.add_scalar('shares_held', np.mean(self.shares_dummy), counter)
			# tb_writer.add_scalar('shares_value', np.mean(self.shares_value_dummy), counter)
			# tb_writer.add_scalar('action', np.mean(self.action_dummy), counter)
			# tb_writer.add_histogram('action_prob', np.mean(self.action_prob_dummy), counter)

	def plot_validation_figures(self, index, name, test_data_label, benchmark):


		if name in ['mean', 'max', 'final']:
			ylimits = [.75 * np.amin(benchmark), 1.5 * np.amax(benchmark)]
		elif name == 'min':
			ylimits = [0., self.settings['inital_account_balance']]
		
		plotcolor = 'darkgreen'

		plt.figure(figsize=(18,18))
		plt.scatter(np.asarray(self.validation_scores)[:,0], np.asarray(self.validation_scores)[:,index])
		plt.grid()
		plt.ylim(ylimits[0], ylimits[1])
		plt.title(name + ' equity statistics over 1 year')
		plt.xlabel('trained episodes')
		plt.ylabel('equity [$]')
		plt.savefig(self.outdir + test_data_label + '/scatter_' + name + '_equity.pdf')
		plt.close()

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
		plt.fill_between(
			np.arange(area_plots.shape[0]), 
			p05, 
			p95, 
			facecolor=plotcolor, 
			alpha=.3)
		plt.fill_between(
			np.arange(area_plots.shape[0]), 
			p25, 
			p75, 
			facecolor=plotcolor, 
			alpha=.8)
		plt.plot(p50, linewidth=3, color='lightblue')
		plt.ylim(ylimits[0], ylimits[1])
		plt.grid()
		plt.title(name + ' equity statistics over 1 year')
		plt.xlabel('trained episodes')
		plt.ylabel('equity [$]')
		plt.savefig(self.outdir + test_data_label + '/area_' + name + '_equity.pdf')
		plt.close()

		plt.figure(figsize=(18,18))
		plt.boxplot(
			box_data, 
			notch=True, 
			labels=None,
			boxprops=dict(color=plotcolor, linewidth=2),
            capprops=dict(color=plotcolor),
            whiskerprops=dict(color=plotcolor),
            flierprops=dict(color=plotcolor, markeredgecolor=plotcolor, markerfacecolor=plotcolor),
            medianprops=dict(color='lightblue', linewidth=2),
            )
		plt.ylim(ylimits[0], ylimits[1])
		plt.grid()
		plt.title('equity statistics over 1 year')
		plt.xlabel('trained episodes')
		plt.ylabel('equity [$]')
		plt.savefig(self.outdir + test_data_label + '/box_' + name + '_equity.pdf')
		plt.close()

	def validate(self, episode, counter, test_data_label):

		try: 
			os.mkdir(self.outdir + test_data_label + '/')
		except Exception: 
			pass

		test_equity = []
		test_trades_buy = []
		test_trades_sell = []

		test_data = self.data['test_' + test_data_label]
		try:
			benchmark = test_data['Close'].values[self.settings['past_horzion']:]
		except KeyError:
			benchmark = test_data['Settle'].values[self.settings['past_horzion']:]
		benchmark /= benchmark[0]
		benchmark *= self.settings['inital_account_balance']

		plt.figure(figsize=(18,18))

		for i in range(0, self.testing_samples):

			if test_data_label == 'gold':
				obs = self.env_test_gold.reset()
			if test_data_label == 'copper':
				obs = self.env_test_copper.reset()
			if test_data_label == 'aluminum':
				obs = self.env_test_aluminum.reset()
			if test_data_label == 'soybean_oil':
				obs = self.env_test_soy_bean.reset()
			if test_data_label == 'dax_futures':
				obs = self.env_test_dax.reset()
			if test_data_label == 'corn':
				obs = self.env_test_corn.reset()
			if test_data_label == 'corn':
				obs = self.env_test_corn.reset()
			if test_data_label == 'canadian_dollar':
				obs = self.env_test_canadian_dollar.reset()

			# obs = self.env_test.reset()

			reward = 0
			done = False
			R = 0
			t = 0


			while not done:

				action = self.agent.act(obs)


				if test_data_label == 'gold':
					obs, reward, done, _, monitor_data = self.env_test_gold.step(action)
				if test_data_label == 'copper':
					obs, reward, done, _, monitor_data = self.env_test_copper.step(action)
				if test_data_label == 'aluminum':
					obs, reward, done, _, monitor_data = self.env_test_aluminum.step(action)
				if test_data_label == 'soybean_oil':
					obs, reward, done, _, monitor_data = self.env_test_soy_bean.step(action)
				if test_data_label == 'dax_futures':
					obs, reward, done, _, monitor_data = self.env_test_dax.step(action)
				if test_data_label == 'corn':
					obs, reward, done, _, monitor_data = self.env_test_corn.step(action)
				if test_data_label == 'canadian_dollar':
					obs, reward, done, _, monitor_data = self.env_test_canadian_dollar.step(action)

				# obs, reward, done, _, monitor_data = self.env_test.step(action)


				test_equity.append(monitor_data['equity'])

				action_choice = np.argmax(softmax(action))
				action_confidence = softmax(action)[action_choice]
				if action_confidence > .8:
					if action_choice == 0:
						test_trades_buy.append([t, monitor_data['equity']])
					if action_choice == 2:
						test_trades_sell.append([t, monitor_data['equity']])

				self.monitor_training(self.writer_test, t, i, done, action, monitor_data, counter)

				R += reward
				t += 1

				if done:
					test_equity = test_equity[:-1]

					plt.plot(test_equity[:-1], linewidth=1)
					# try:
					# 	plt.scatter(np.asarray(test_trades_buy)[:,0], np.asarray(test_trades_buy)[:,1], marker='X', c='green', s=5)
					# 	plt.scatter(np.asarray(test_trades_sell)[:,0], np.asarray(test_trades_sell)[:,1], marker='X', c='red', s=5)
					# except IndexError:
					# 	pass

					self.validation_scores.append([counter, np.mean(test_equity), np.amin(test_equity), np.amax(test_equity), test_equity[-1]])
					test_equity = []

					self.agent.stop_episode()


		time_axis = test_data.index[self.settings['past_horzion']:].date
		time_axis_short = time_axis[::10]

		plt.plot(benchmark , linewidth=3, color='k', label='close')
		plt.ylim(.75 * np.amin(benchmark), 1.5 * np.amax(benchmark))
		plt.xticks(np.linspace(0, len(time_axis), len(time_axis_short)-1), time_axis_short, rotation=90)
		plt.grid()
		plt.title(test_data_label + ' validation runs at episode ' + str(episode))
		plt.xlabel('episode')
		plt.ylabel('equity [$]')
		plt.legend()
		plt.savefig(self.outdir + test_data_label + '/validation_E' + str(episode) + '.pdf')
		plt.close()

		self.plot_validation_figures(1, 'mean', test_data_label, benchmark)
		self.plot_validation_figures(2, 'min', test_data_label, benchmark)
		self.plot_validation_figures(3, 'max', test_data_label, benchmark)
		self.plot_validation_figures(4, 'final', test_data_label, benchmark)

	def train(self):

		print('\nstart training loop\n')

		def check_types(input, inputname):
			if np.isnan(input).any(): print('----> ', inputname, ' array contains NaN\n', np.isnan(input).shape, '\n')
			if np.isinf(input).any(): print('----> ', inputname, ' array contains inf\n', np.isinf(input).shape, '\n')


		n_episodes = int(1e5)

		log_data = []
		action_log = []

		debug_printing = False


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
					action_log.append([self.training_counter, action[0], action[1], action[2]])
			
			
				if done:
					if i % 10 == 0:
						print('\nrollout ' + str(i) + '\n', pd.DataFrame(log_data).max())
					log_data = []
					self.training_scores.append([i, R])
					self.training_counter += 1
			
			self.agent.stop_episode()

			if i % self.monitor_freq == 0:

				# self.agent.stop_episode_and_train(obs, reward, done)

				# print('\n\nvalidation...')
				self.validate(i, self.training_counter, 'gold')
				if debug_printing: print('\n\n****************\nSOY BEANS\n\n')
				self.validate(i, self.training_counter, 'soybean_oil')
				if debug_printing: print('\n\n****************\nCORN\n\n')
				self.validate(i, self.training_counter, 'corn')
				# if debug_printing: print('\n\n****************\nCANADIAN DOLLAR\n\n')
				# self.validate(i, self.training_counter, 'canadian_dollar')

				if debug_printing: print('\n****************\n')

				act_probs = softmax(np.asarray(action_log)[:,1:], axis=1)

				plt.figure()
				plt.scatter(np.asarray(self.training_scores)[:,0], np.asarray(self.training_scores)[:,1], s=2, label='reward')
				plt.legend()
				plt.title('reward')
				plt.grid()
				plt.savefig(self.outdir + 'reward.pdf')
				plt.close()
			
				plt.figure()
				plt.scatter(np.asarray(action_log)[:,0], act_probs[:,0], label='action0')
				plt.scatter(np.asarray(action_log)[:,0], act_probs[:,1], label='action1')
				plt.scatter(np.asarray(action_log)[:,0], act_probs[:,2], label='action2')
				plt.legend()
				plt.title('actions')
				plt.grid()
				plt.savefig(self.outdir + 'actions.pdf')
				plt.close()
			
				plt.figure()
				plt.plot(np.asarray(action_log)[:,0], act_probs[:,0], label='action0')
				plt.plot(np.asarray(action_log)[:,0], act_probs[:,1], label='action1')
				plt.plot(np.asarray(action_log)[:,0], act_probs[:,2], label='action2')
				plt.legend()
				plt.title('actions')
				plt.grid()
				plt.savefig(self.outdir + 'actions_plot.pdf')
				plt.close()


			if i % 10 == 0 and i > 0:

				self.agent.save(self.outdir)

				serializers.save_npz(self.outdir + 'model.npz', self.model)

			
			# if i % 1000 == 0:
			#     print('\nepisode:', i, ' | episode length: ', t, '\nreward:', R,
			#           '\nstatistics:', self.agent.get_statistics(), '\n')

		self.agent.stop_episode_and_train(obs, reward, done)
		print('Finished.')


rl_stock_trader().train()


