# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-05-28 14:45:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-26 12:36:19

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import ode, odeint, solve_ivp
from tqdm import tqdm

from .utils import *
from .logger import *


class ODESolver:
    ''' Generic interface to ODE solver object. '''

    def __init__(self, ykeys, dfunc, dt=None):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivative function
            :param dt: integration time step (s)
        '''
        self.ykeys = ykeys
        self.dfunc = dfunc
        self.dt = dt

    def check_func(self, key, value):
        if not callable(value):
            raise ValueError(f'{key} function must be a callable object')

    @property
    def ykeys(self):
        return self._ykeys

    @ykeys.setter
    def ykeys(self, value):
        if not is_iterable(value):
            value = list(value)
        for item in value:
            if not isinstance(item, str):
                raise ValueError('ykeys must be a list of strings')
        self._ykeys = value

    @property
    def nvars(self):
        return len(self.ykeys)

    @property
    def dfunc(self):
        return self._dfunc

    @dfunc.setter
    def dfunc(self, value):
        self.check_func('derivative', value)
        self._dfunc = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value is None:
            self._dt = None
        else:
            if not isinstance(value, float):
                raise ValueError('time step must be float-typed')
            if value <= 0:
                raise ValueError('time step must be strictly positive')
            self._dt = value

    def get_nsamples(self, t0, tend, dt=None):
        ''' Get the number of samples required to integrate across 2 times with a given time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :param dt: integration time step (s)
            :return: number of required samples, rounded to nearest integer
        '''
        if dt is None:
            dt = self.dt
        return max(int(np.round((tend - t0) / dt)), 2)

    def get_time_vector(self, t0, tend, **kwargs):
        ''' Get the time vector required to integrate from an initial to a final time with
            a specific time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :return: vector going from current time to target time with appropriate step (s)
        '''
        return np.linspace(t0, tend, self.get_nsamples(t0, tend, **kwargs))

    def initialize(self, y0, t0=0.):
        ''' Initialize global time vector, state vector and solution array.

            :param y0: dictionary of initial conditions
            :param t0: optional initial time or time vector (s)
        '''
        keys = list(y0.keys())
        if len(keys) != len(self.ykeys):
            raise ValueError("Initial conditions do not match system's dimensions")
        for k in keys:
            if k not in self.ykeys:
                raise ValueError(f'{k} is not a differential variable')
        y0 = {k: np.asarray(v) if is_iterable(v) else np.array([v]) for k, v in y0.items()}
        ref_size = y0[keys[0]].size
        if not all(v.size == ref_size for v in y0.values()):
            raise ValueError('dimensions of initial conditions are inconsistent')
        self.y = np.array(list(y0.values())).T
        self.t = np.ones(self.y.shape[0]) * t0
        if hasattr(self, 'xref'):
            self.x = np.zeros(self.t.size)

    def append(self, t, y):
        ''' Append to global time vector, state vector and solution array.

            :param t: new time vector to append (s)
            :param y: new solution matrix to append
        '''
        self.t = np.concatenate((self.t, t))
        self.y = np.concatenate((self.y, y), axis=0)
        if hasattr(self, 'x'):
            self.x = np.concatenate((self.x, np.ones(t.size) * self.xref))

    def bound(self, tbounds):
        ''' Restrict global time vector, state vector ans solution matrix within
            specific time range.

            :param tbounds: minimal and maximal allowed time restricting the global arrays (s).
        '''
        i_bounded = np.logical_and(self.t >= tbounds[0], self.t <= tbounds[1])
        self.t = self.t[i_bounded]
        self.y = self.y[i_bounded, :]
        if hasattr(self, 'x'):
            self.x = self.x[i_bounded]

    @staticmethod
    def time_str(t):
        return f'{t * 1e3:.5f} ms'

    def timedlog(self, s, t=None):
        ''' Add preceding time information to log string. '''
        if t is None:
            t = self.t[-1]
        return f't = {self.time_str(t)}: {s}'

    def integrate_until(self, target_t, remove_first=False):
        ''' Integrate system until a target time and append new arrays to global arrays.

            :param target_t: target time (s)
            :param remove_first: optional boolean specifying whether to remove the first index
            of the new arrays before appending
        '''
        if target_t < self.t[-1]:
            raise ValueError(f'target time ({target_t} s) precedes current time {self.t[-1]} s')
        elif target_t == self.t[-1]:
            t, y = self.t[-1], self.y[-1]
        if self.dt is None:
            sol = solve_ivp(
                self.dfunc, [self.t[-1], target_t], self.y[-1], method='LSODA')
            t, y = sol.t, sol.y.T
        else:
            t = self.get_time_vector(self.t[-1], target_t)
            y = odeint(self.dfunc, self.y[-1], t, tfirst=True)
        if remove_first:
            t, y = t[1:], y[1:]
        self.append(t, y)

    def resample_arrays(self, t, y, target_dt):
        ''' Resample a time vector and soluton matrix to target time step.

            :param t: time vector to resample (s)
            :param y: solution matrix to resample
            :param target_dt: target time step (s)
            :return: resampled time vector and solution matrix
        '''
        tnew = self.get_time_vector(t[0], t[-1], dt=target_dt)
        ynew = np.array([np.interp(tnew, t, x) for x in y.T]).T
        return tnew, ynew

    def resample(self, target_dt):
        ''' Resample global arrays to a new target time step.

            :param target_dt: target time step (s)
        '''
        tnew, self.y = self.resample_arrays(self.t, self.y, target_dt)
        if hasattr(self, 'x'):
            self.x = interp1d(self.t, self.x, kind='nearest', assume_sorted=True)(tnew)
        self.t = tnew

    def solve(self, y0, tstop, **kwargs):
        ''' Simulate system for a given time interval for specific initial conditions.

            :param y0: dictionary of initial conditions
            :param tstop: stopping time (s)
        '''
        # Initialize system
        self.initialize(y0, **kwargs)

        # Integrate until tstop
        self.integrate_until(tstop, remove_first=True)

    @property
    def solution(self):
        ''' Return solution as a pandas dataframe.

            :return: timeseries dataframe with labeled time, state and variables vectors.
        '''
        df = pd.DataFrame({k: self.y[:, i] for i, k in enumerate(self.ykeys)})
        df['t'] = self.t
        if hasattr(self, 'x'):
            df['x'] = self.x
        df.set_index('t', inplace=True)
        return df

    def __call__(self, *args, target_dt=None, max_nsamples=None, **kwargs):
        ''' Specific call method: solve the system, resample solution if needed, and return
            solution dataframe. '''
        self.solve(*args, **kwargs)
        if target_dt is not None:
            self.resample(target_dt)
        elif max_nsamples is not None and self.t.size > max_nsamples:
            self.resample(np.ptp(self.t) / max_nsamples)
        return self.solution


class EventDrivenSolver(ODESolver):
    ''' Event-driven ODE solver. '''

    def __init__(self, eventfunc, *args, event_params=None, **kwargs):
        ''' Initialization.

            :param eventfunc: function called on each event
            :param event_params: dictionary of parameters used by the derivatives function
        '''
        super().__init__(*args, **kwargs)
        self.eventfunc = eventfunc
        self.assign_event_params(event_params)

    def assign_event_params(self, event_params):
        ''' Assign event parameters as instance attributes. '''
        if event_params is not None:
            for k, v in event_params.items():
                setattr(self, k, v)

    @property
    def eventfunc(self):
        return self._eventfunc

    @eventfunc.setter
    def eventfunc(self, value):
        self.check_func('event', value)
        self._eventfunc = value

    @property
    def xref(self):
        return self._xref

    @xref.setter
    def xref(self, value):
        self._xref = value

    def initialize(self, *args, **kwargs):
        self.xref = 0
        super().initialize(*args, **kwargs)

    def fire_event(self, xevent):
        ''' Call event function and set new xref value. '''
        if xevent is not None:
            if xevent == 'log':
                self.log_progress()
            else:
                self.eventfunc(xevent)
                self.xref = xevent

    def init_log(self, logfunc, n):
        ''' Initialize progress logger. '''
        self.logfunc = logfunc
        if self.logfunc is None:
            assign_log_handler(logger, TqdmHandler(my_log_formatter))
            self.pbar = tqdm(total=n)
        else:
            self.np = n
            logger.debug('integrating stimulus')

    def log_progress(self):
        ''' Log simulation progress. '''
        if self.logfunc is None:
            self.pbar.update()
        else:
            logger.debug(self.timedlog(self.logfunc(self.y[-1])))

    def terminate_log(self):
        ''' Terminate progress logger. '''
        if self.logfunc is None:
            self.pbar.close()
        else:
            logger.debug('integration completed')

    def sort_events(self, events):
        ''' Sort events pairs by occurence time. '''
        return sorted(events, key=lambda x: x[0])

    def solve(self, y0, events, tstop, log_period=None, logfunc=None, **kwargs):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param events: list of events
            :param tstop: stopping time (s)
        '''
        # Sort events according to occurrence time
        events = self.sort_events(events)

        # Make sure all events occur before tstop
        if events[-1][0] > tstop:
            raise ValueError('all events must occur before stopping time')

        if log_period is not None:  # Add log events if any
            tlogs = np.arange(kwargs.get('t0', 0.), tstop, log_period)[1:]
            if tstop not in tlogs:
                tlogs = np.hstack((tlogs, [tstop]))
            events = self.sort_events(events + [(t, 'log') for t in tlogs])
            self.init_log(logfunc, tlogs.size)
        else:  # Otherwise, add None event at tstop
            events.append((tstop, None))

        # Initialize system
        self.initialize(y0, **kwargs)

        # For each upcoming event
        for i, (tevent, xevent) in enumerate(events):
            self.integrate_until(  # integrate until event time
                tevent,
                remove_first=i > 0 and events[i - 1][1] == 'log')
            self.fire_event(xevent)  # fire event

        # Terminate log if any
        if log_period is not None:
            self.terminate_log()
