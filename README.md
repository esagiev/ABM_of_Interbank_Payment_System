The version on 20 July 2020.

NOTICE:  All information contained herein is the property of Erkin Sagiev.
    Dissemination of this information or reproduction of this material
    is strictly forbidden unless prior written permission is obtained
    from Erkin Sagiev.

### The Agent Based Model of Interbank Payment System

The model includes a set of banks, which receive payment requests from their customers
    and execute these requests at time period they prefer the most. If time of execution is
    is differ from time of arrival, bank incurs deferral costs. There can be 3 types of
    the banks. Each type has own average value of payment request.

The average request is used as a rate parameter for Poisson distribution to sample
    payment requests. Payee for each payment request is sampled from PMF distribution.

The global parameters of the model:

- daytime interest rate;
- overnight interest
- daytime deferral cost rate;
- overnight deferral cost;
- array of bank types;
- array of number of banks of each type;
- PMF for distribution of payees;
- number of time periods during the day;
- length of bank's memory.

The result of simulations is a figure with average cumulative share of executed payments in
total value of requests. The example for the case, when deferral costs are larger than
interest costs, is provided below (first 50 time periods).

![Deferral dynamics](/high_defrate.png)

Another result of simulation is a network, where a link is formed, if all payment request to
specific bank-payee are executed by the end of the day. The example for the case, when
interest costs are very high is provided below.

![Network](/network.png)
