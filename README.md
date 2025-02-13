# Alert Email Filtering

## About

This Python project was designed filter out repetitive emails received from several services across a myriad of servers to prevent alert fatigue. It also allows new users to be added and removed from alerts in a centralized location.

Alerts being sent by services should be pointed to the given SMTP new address. Every few minutes, the new emails received will be run through a few simple machine learning models that determine by majority quorum whether or not the email is a useful or useless alert. The heuristic we operate under is that any extremely novel and unique alert may be important, so never-encountered warnings will always make it through.

Since new machines producing alerts are set up all the time, this project supports a dynamic user feedback system and training-in-place to keep the ML model fresh. Replying with the keywords [STOP, UNSUBSCRIBE, USELESS, BAD, NO] to an alert email forwarded to you will prompt the model to mark it as a useless email for future forwarding and also block future similar alerts from reaching you.

Conversely, to see more consistent results, reply to useful alerts with [SUBSCRIBE, USEFUL, GOOD, YES] to keep receiving them and better inform the model how to differentiate between a useful and useless alert. This decision, if the email is later recognized as spam, can always be overriden later.

## For Future Improvement

- Integrate with open source SMTP mailbox so the receiving address can be easily scraped from and swapped out.
- Improve similarity hashing to be more tolerant of different-length emails. See the [old Gmail spam detection](https://en.wikipedia.org/wiki/SimHash#:~:text=In%20computer%20science%2C%20SimHash%20is,was%20created%20by%20Moses%20Charikar.) algorithm SimHash: https://github.com/1e0ng/simhash?tab=readme-ov-file
