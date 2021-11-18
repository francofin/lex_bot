### Required Libraries ###
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import pandas as pd
from botocore.vendored import requests
import numpy as np
import json
from datetime import date


api_key = os.environ.get('FMP_API')
today = date.today()
date_time = today.strftime("%Y-%m-%d")

### Functionality Helper Functions ###
def parse_int(n):
    """
    Securely converts a non-integer value to integer.
    """
    try:
        return int(n)
    except ValueError:
        return float("nan")


def build_validation_result(is_valid, violated_slot, message_content):
    """
    Define a result message structured as Lex response.
    """
    if message_content is None:
        return {"isValid": is_valid, "violatedSlot": violated_slot}

    return {
        "isValid": is_valid,
        "violatedSlot": violated_slot,
        "message": {"contentType": "PlainText", "content": message_content},
    }

def validate_data(age, investment_amount, intent_request):
    """
    Validates the data provided by the user.
    """

    # Validate that the user is over 18 years old
    if age is not None:
        if (age > 65) and (age < 18):
            return build_validation_result(
                False,
                "age",
                "You should be at least 18 years old to use this service, "
                "please have someone else look at your portfolio",
            )

    # Validate the investment amount, it should be > 0
    if investment_amount is not None:
        investment_amount = parse_int(
            investment_amount
        )  # Since parameters are strings it's important to cast values
        if investment_amount <= 5000:
            return build_validation_result(
                False,
                "investment_amount",
                "You need more than 5000 to invest",
            )
            

    # A True results is returned if age or amount are valid
    return build_validation_result(True, None, None)

### Dialog Actions Helper Functions ###
def get_slots(intent_request):
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]


def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    """
    Defines an elicit slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "ElicitSlot",
            "intentName": intent_name,
            "slots": slots,
            "slotToElicit": slot_to_elicit,
            "message": message,
        },
    }


def delegate(session_attributes, slots):
    """
    Defines a delegate slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": "Delegate", "slots": slots},
    }


def close(session_attributes, fulfillment_state, message):
    """
    Defines a close slot type response.
    """

    response = {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": fulfillment_state,
            "message": message,
        },
    }

    return response

def get_investment_recommendation(risk_level):
    investment_options = ['USMV', 'SPLV', 'ONEV', 'SPYG', 'JMOM', 'VUG', 'IWF', 'QQQ', 'IVW', 'IWO', 'VTWG', 'BITO']

    all_data = []
    for stock in investment_options:
        data = (pd.DataFrame(json.loads(requests.get(f'https://fmpcloud.io/api/v3/historical-price-full/'+stock+'?from=2020-01-01&to='+date_time+'&apikey='+api_key).content)['historical']).set_index('date').iloc[::-1])['close'].to_frame().rename(columns={'close':stock})
        all_data.append(data)
    total_df = pd.concat(all_data, axis=1)
    daily_returns = total_df.pct_change().fillna(0)
    covariance = daily_returns.cov()*252

    num_ports = 1000

    all_weights = np.zeros((num_ports,len(daily_returns.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):

        # Create Random Weights
        weights = np.array(np.random.random(len(daily_returns.columns)))

        # Rebalance Weights
        weights = weights / np.sum(weights)
        
        # Save Weights
        all_weights[ind,:] = weights

        # Expected Return
        ret_arr[ind] = np.sum((daily_returns.mean() * weights) *252)

        # Expected Variance
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

    if risk_level =='Low':
        portfolio_sharpe = sharpe_arr.max()
        expected_return = ret_arr[sharpe_arr.argmax()]
        port_weights = all_weights[sharpe_arr.argmax(),:]
        recommended_port = list(zip(list(round(port_weights*100,2)),investment_options))
    elif risk_level =='Medium':
        portfolio_selected = len(sharpe_arr)/2
        portfolio_sharpe = sharpe_arr[portfolio_selected]
        expected_return = ret_arr[portfolio_selected]
        port_weights = all_weights[portfolio_selected,:]
        recommended_port = list(zip(list(round(port_weights*100,2)),investment_options))
    elif risk_level =='High':
        portfolio_selected = int(len(ret_arr)/1.05)
        portfolio_sharpe = sharpe_arr[portfolio_selected]
        expected_return = ret_arr[portfolio_selected]
        port_weights = all_weights[portfolio_selected,:]
        recommended_port = list(zip(list(round(port_weights*100,2)),investment_options))
    elif risk_level =='Maximum':
        portfolio_sharpe = sharpe_arr[ret_arr.argmax()]
        expected_return = ret_arr.max()
        port_weights = all_weights[ret_arr.argmax(),:]
        recommended_port = list(zip(list(round(port_weights*100,2)),investment_options))

    return expected_return, recommended_port


### Intents Handlers ###
def recommend_portfolio(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    first_name = get_slots(intent_request)["firstName"]
    age = get_slots(intent_request)["age"]
    investment_amount = get_slots(intent_request)["investmentAmount"]
    risk_level = get_slots(intent_request)["riskLevel"]
    source = intent_request["invocationSource"]

    if source == "DialogCodeHook":
        # Perform basic validation on the supplied input slots.
        slots = get_slots(intent_request)
        # Use the elicitSlot dialog action to re-prompt
        # for the first violation detected.

        ### YOUR DATA VALIDATION CODE STARTS HERE ###
        validation_result = validate_data(age, investment_amount, intent_request)

        # If the data provided by the user is not valid,
        # the elicitSlot dialog action is used to re-prompt for the first violation detected.
        if not validation_result["isValid"]:
            slots[validation_result["violatedSlot"]] = None  # Cleans invalid slot

            # Returns an elicitSlot dialog to request new data for the invalid slot
            return elicit_slot(
                intent_request["sessionAttributes"],
                intent_request["currentIntent"]["name"],
                slots,
                validation_result["violatedSlot"],
                validation_result["message"],
            )
        ### YOUR DATA VALIDATION CODE ENDS HERE ###

        # Fetch current session attibutes
        output_session_attributes = intent_request["sessionAttributes"]

        return delegate(output_session_attributes, get_slots(intent_request))

    # Get the initial investment recommendation
    
    ### YOUR FINAL INVESTMENT RECOMMENDATION CODE STARTS HERE ###
    expected_return, recommended_portfolio = get_investment_recommendation(risk_level)
    ### YOUR FINAL INVESTMENT RECOMMENDATION CODE ENDS HERE ###

    # Return a message with the initial recommendation based on the risk level.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": """{} thank you for your information; 
            based on the risk level you defined, my recommendation is to choose an investment portfolio with 
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            {}
            Your expected return can be {}
            Past returns are not representative of future returns
            """.format(
                first_name, recommended_portfolio[0], recommended_portfolio[1], recommended_portfolio[2], recommended_portfolio[3], recommended_portfolio[4], recommended_portfolio[5], recommended_portfolio[6], recommended_portfolio[7], 
                recommended_portfolio[8], recommended_portfolio[9], recommended_portfolio[10], recommended_portfolio[11], expected_return
            ),
        },
    )


### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "RecommendPortfolio":
        return recommend_portfolio(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)
