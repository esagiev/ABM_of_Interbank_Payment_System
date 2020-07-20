# The Agent-Based Model of Interbank Payment System.
# Version on 19 July 2020.


class Agent():
    """ The class creates a bank-agent of a type, namely 0,1,2,
    regarding its average value of payment request. Each bank-agent has
    personal variables that track its beliefs, queues, costs,
    and liquidity.
    """
    def __init__(self, b_type):
        self.type = b_type
        self.x_avr = type_v[b_type]
        self.belief_in = self.x_avr   # Starting beliefs.
        self.belief_out = self.x_avr
        self.defcost = 0   # Starting costs.
        self.intcost = 0
        self.liqpos = self.x_avr   # Starting liquidity position.
        self.memory_in = collections.deque(maxlen=max_length)
        self.memory_out = collections.deque(maxlen=day_length)
        self.queue = list([0] for _ in range(sum(n)))
    
    def belief_update(self, t_cur):
        """ Defines bank's belief on future incoming payments (in)
        and outgoing payment requests (out).
        """
        self.belief_in = (sum(self.memory_in)+self.x_avr)*1/t_cur +\
			1/(2*t_cur)
        self.belief_out = (sum(self.memory_out)+self.x_avr)*1/t_cur +\
			1/(2*t_cur)
    
    def defcost_add(self):
        """ Calculates and sums up deferral costs, which should be
        negative. Should be used once per time period.
        """
        queue = np.sum(np.sum(self.queue))
        self.defcost = self.defcost - d_rate*queue
    
    def intcost_add(self):
        """ Calculates and sums up interests on current net liquidity
        position. Should be used once per time period.
        """
        self.intcost = self.intcost + i_rate*self.liqpos


class Payment():
    """ The class generates sequence of incoming payment request
    for all types of bank-agent simultaneously. Three types only.
    """
    def __init__(self):
        self.p_matrix = np.zeros([len(n),sum(n)])
        self.sample = np.zeros([2,sum(n)],dtype=int)
        self.id = range(sum(n))
    
    def make_rvs(self):
        """ Creates PMF for each bank-type to sample payees. """
        for t in range(len(n)):
            self.p_matrix[t,:n[0]] = p_table[t,0]/n[0]
            self.p_matrix[t,n[0]:(n[0]+n[1])] = p_table[t,1]/n[1]
            self.p_matrix[t,(n[0]+n[1]):] = p_table[t,2]/n[2]
        
        rvs_0 = stats.rv_discrete(values=(self.id, self.p_matrix[0,:]))
        rvs_1 = stats.rv_discrete(values=(self.id, self.p_matrix[1,:]))
        rvs_2 = stats.rv_discrete(values=(self.id, self.p_matrix[2,:]))
        
        return rvs_0, rvs_1, rvs_2
    
    def sampling(self, payee_rvs):
        """ The callable method to sample all payment requests. """
        payee_0, payee_1, payee_2 = payee_rvs
        
        # Generating payee for each bank-payer.
        self.sample[0,:n[0]] = payee_0.rvs(size=n[0])
        self.sample[0,n[0]:(n[0]+n[1])] = payee_1.rvs(size=n[1])
        self.sample[0,(n[0]+n[1]):] = payee_2.rvs(size=n[2])
        
        # Generating value for each payment request.
        self.sample[1,:n[0]] =\
			stats.poisson.rvs(mu=type_v[0], size=n[0])
        self.sample[1,n[0]:(n[0]+n[1])] =\
			stats.poisson.rvs(mu=type_v[1], size=n[1])
        self.sample[1,(n[0]+n[1]):] =\
			stats.poisson.rvs(mu=type_v[2], size=n[2])
        
        return self.sample


class Market():
    """ The class of netting mechanism of counter payment claims. """
    def __init__(self):
        self.matrix = np.zeros([sum(n),sum(n)])
    
    def reset(self):
        """ Cleaning the matrix for next time period. """
        self.matrix = np.zeros([sum(n),sum(n)])
    
    def add_payment(self, payer_id, queue_cut):
        """ Sending the payment request for execution. """
        self.matrix[payer_id,:] = queue_cut
    
    def net_liqpos(self, bank_id):
        """ Returns result of netting. If postitive, net borrowing.
        If negative, provision of deposit on the market. Real
        borrowings or deposits are depend on liqudity position.
        """
        sum_in = np.sum(self.matrix,0)[bank_id]
        sum_out = np.sum(self.matrix,1)[bank_id]
        sum_dif = sum_in - sum_out # Negative is borrowing.
        
        return sum_in, sum_out, sum_dif


class Trace():
    """ Collects information on payments and bank-agents. In
    pay_trace matrix 3rd dim: 0 is value of payment requests,
    1 is number of payment requests, 2 is value of ougoing payments,
    3 is number of ougoing payments. """
    def __init__(self):
        self.pay_trace = np.zeros([sum(n),sum(n),4])
        self.def_rate = np.zeros([sum(n),day_length])
        self.vals = np.zeros([2,sum(n)])
    
    def tracking(self, payments, market, time):
        # Tracking all sampled payment requests.
        index_0 = (range(sum(n)), payments[0,:],\
				   np.zeros(sum(n),dtype=int))
        value_0 = payments[1,:]
        self.pay_trace[index_0] += value_0
        
        # Tracking number of sampled payment requests.
        index_1 = (range(sum(n)), payments[0,:],\
				   np.ones(sum(n),dtype=int))
        value_1 = payments[1,:] != 0
        self.pay_trace[index_1] += value_1
        
        # Tracking factual outgoing payments.
        market_mat = market.matrix
        self.pay_trace[:,:,2] += market_mat
        
        # Tracking number of factual outgoing payments.
        value_3 = market_mat != 0
        self.pay_trace[:,:,3] += value_3
        
        # Tracking share of accumulated deferred payment at each time.
        self.vals[0,:] += payments[1,:]
        self.vals[1,:] += np.sum(market_mat,1)
        for b in range(sum(n)):
            if self.vals[0,b] != 0:
                self.def_rate[b,time] = self.vals[1,b]/self.vals[0,b]


def defer(bank, t_cur):
    """ Compares total value of payment for current and next time
    period, then chooses deferral removing the largest request.
    """
    queue = bank.queue
    queue_cut = np.zeros(sum(n))
    value_dif = value(bank, t_cur, 1) - value(bank, t_cur, 0)
    while True:
        if value_dif > 0:
            max_list = []
            [max_list.append(max(p)) for p in queue]
            max_value = max(max_list)
            if max_value == 0:
                break
            max_index = max_list.index(max_value)
            queue_cut[max_index] += max_value
            queue[max_index].remove(max_value)
        else:
            break
    
    return queue_cut


def run_model():
    """ This class runs the agent-based model as one piece. """
    if len(n) != 3 or len(type_v) != 3:
        raise Exception('Number of bank types must be equal to 3.')
    
    # Creating the list of bank-agents.
    banks_types = np.asarray([],dtype=int)
    for t in range(len(n)):
        types = np.ones(n[t],dtype=int)*t
        banks_types = np.concatenate((banks_types, types))
    
    banks = list()
    for b in banks_types:
        banks.append(Agent(b))
    
    # Creating market, payment sampler, and tracking.
    market = Market()
    sampler = Payment()
    trace = Trace()
    payee_rvs = sampler.make_rvs()
    
    # The main loop.
    for time in range(day_length):
        time_cur = time +1
        
        # Update of beliefs based on payment history.
        for b in range(sum(n)):
            banks[b].belief_update(time_cur)
        
        # Sampling of pament requests.
        payments = sampler.sampling(payee_rvs)

        # Making deferral decision for each bank.
        for b in range(sum(n)):
            # Queue and memory of outgoing payments update.
            banks[b].queue[payments[0,b]].append(payments[1,b])
            banks[b].memory_out.append(payments[1,b])
            queue_cut = defer(banks[b], time) # Deferral and netting.
            market.add_payment(b, queue_cut)
        
        # Costs and memory of incoming payments update.
        for b in range(sum(n)):
            pay_in, _, liq_dif = market.net_liqpos(b)
            banks[b].memory_in.append(pay_in)
            banks[b].liqpos += liq_dif
            banks[b].defcost_add()
            banks[b].intcost_add()
            
        # Tracking and market reset for new period.
        trace.tracking(payments, market, time)
        market.reset()
    
    return banks, trace


def value(bank, t_given, add):
    """ Calculates total cost of the queue for the whole day, which
    inludes factual cost upto current time and future possible costs.
    """
    queue = sum(np.sum(bank.queue))
    belief_dif = bank.belief_in - bank.belief_out
    
    # Queue updates for possible future, only if it increases.
    if belief_dif < 0:
        queue += add*belief_dif
        
    # Optimal deferral period.
    if belief_dif > 0 and queue/belief_dif > 1/2:
        t_opt = np.ceil(queue/belief_dif -1/2)
    else:
        t_opt = 0
    
    # Getting deferral period within the daytime left.
    if t_given+add + t_opt >= day_length:
        t = day_length - t_given+add
    else:
        t = t_opt
    
    # Daytime left after repaying the queue.
    day_left = day_length - t_given+add +1
    t_left = day_left - t
    
    # Making copies, becuase changes are not factual.
    defcost = copy.copy(bank.defcost)
    intcost = copy.copy(bank.intcost)
    liqpos = copy.copy(bank.liqpos)
    
    # Adjustment copies for the time of evaluation.
    liqpos += add*belief_dif
    intcost += add*i_rate*liqpos
    defcost += add*d_rate*queue
    
    # Future costs during repayment of the queue and left daytime.
    unpaid = t*queue - t*(t+1)*belief_dif/2
    liqleft = liqpos + t_left*(t_left+1)*belief_dif/2
    
    if unpaid <= 0:
        future_defcost = -d_rate*t*(t+1)*belief_dif/2
        future_interest = i_rate*day_left*liqpos +\
			i_rate*t_left*(t_left+1)*belief_dif/2 +\
			i_night*liqleft
    else:
        future_defcost = -d_rate*t*(t+1)*belief_dif/2 -\
			d_rate*day_left*unpaid - d_night*unpaid
        future_interest = i_rate*day_left*liqpos + i_night*liqpos
    
    # Deriving full cost of payment requests.
    return intcost + defcost + future_defcost + future_interest
