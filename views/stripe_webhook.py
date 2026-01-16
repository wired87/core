import json
import os
import stripe
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from fb_core.firestore_manager import FirestoreMgr

stripe.api_key = os.environ.get("STRIPE_API_KEY")
endpoint_secret = os.environ.get("STRIPE_ENDPOINT_SECRET")

firestore = FirestoreMgr()

# Map Price IDs to Plans
PRICE_MAP = {
    os.environ.get("PRICE_MAGICIAN", "price_1Qj..."): "magician",
    os.environ.get("PRICE_WIZARD", "price_1Qk..."): "wizard"
}

@csrf_exempt
@require_POST
def stripe_webhook(request):
    payload = request.body
    sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        return HttpResponse(status=400)
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        return HttpResponse(status=400)

    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['admin_data']['object']
        handle_checkout_session_completed(session)
    elif event['type'] == 'customer.subscription.updated':
        subscription = event['admin_data']['object']
        handle_subscription_updated(subscription)
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['admin_data']['object']
        handle_subscription_deleted(subscription)
    elif event['type'] == 'invoice.payment_succeeded':
        invoice = event['admin_data']['object']
        handle_invoice_payment_succeeded(invoice)
    else:
        print('Unhandled event type {}'.format(event['type']))

    return JsonResponse({'status': 'success'})

def handle_checkout_session_completed(session):
    user_id = session.get("metadata", {}).get("user_id")
    if not user_id:
        user_id = session.get("client_reference_id")
    
    if not user_id:
        print("No user_id found in session")
        return

    # Retrieve line items to find the price/plan
    # Note: In a real scenario, we might need to expand line_items or fetch the subscription
    # For simplicity, we'll assume we can get the plan from the subscription or metadata if passed
    # But here, let's try to fetch subscription to get the plan
    subscription_id = session.get("subscription")
    if subscription_id:
        try:
            sub = stripe.Subscription.retrieve(subscription_id)
            price_id = sub['items']['admin_data'][0]['price']['id']
            plan = PRICE_MAP.get(price_id, "magician") # Default to magician if unknown
            
            print(f"Upgrading user {user_id} to {plan}")
            
            # Update Firestore
            user_doc = firestore.get_user_doc(user_id)
            if not user_doc:
                firestore.create_default_user(user_id)
            
            # Update plan
            firestore.update_user_plan(user_id, plan)
        except Exception as e:
            print(f"Error processing subscription: {e}")

def handle_subscription_updated(subscription):
    # Check status
    status = subscription['status']
    user_id = subscription.get("metadata", {}).get("user_id")
    
    # If metadata is missing on subscription (it might be on the customer or session), 
    # we might need to look up the user by stripe_customer_id if we stored it.
    # For this implementation, we assume metadata is propagated or we have a mapping.
    if not user_id:
        print("No user_id in subscription update")
        return

    if status == 'active':
        price_id = subscription['items']['admin_data'][0]['price']['id']
        plan = PRICE_MAP.get(price_id, "magician")
        print(f"Subscription active for {user_id}, plan: {plan}")
        firestore.update_user_plan(user_id, plan)
    elif status in ['past_due', 'unpaid', 'canceled']:
        print(f"Subscription {status} for {user_id}, downgrading to free")
        firestore.update_user_plan(user_id, "free")

def handle_subscription_deleted(subscription):
    user_id = subscription.get("metadata", {}).get("user_id")
    if user_id:
        print(f"Subscription deleted for {user_id}, downgrading to free")
        firestore.update_user_plan(user_id, "free")

def handle_invoice_payment_succeeded(invoice):
    # Maybe add some compute credits as a bonus?
    pass
