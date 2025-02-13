# Alert Email Filtering

## About

We receive tons of alerts from a mountain of servers in Chicago. It would be nice to filter out the repetitive ones to prevent alert fatigue and also allow new users to be added and removed from alerts in a central location. This project seeks to accomplish all that.

Alerts should be pointed to the new address [TODO: get one and put it here]. Every few minutes, the new emails received will be run through a simple machine learning model that determines whether or not the email is a useful or useless alert. The heuristic we operate under is that any completely new and unique alert may be important, so never-encountered warnings will always make it through.

Since new machines producing alerts are set up all the time, this project supports a dynamic user feedback system and training-in-place to keep the ML model fresh. Replying with the keywords [STOP, UNSUBSCRIBE, USELESS, BAD, NO] to an alert email forwarded to you will prompt the model to mark it as a useless email for future forwarding and also block future similar alerts from reaching you.

Conversely, to see more consistent results, reply to useful alerts with [SUBSCRIBE, USEFUL, GOOD, YES] to keep receiving them and better inform the model how to differentiate between a useful and useless alert. This decision, if it becomes spammy, can always be overriden later.
