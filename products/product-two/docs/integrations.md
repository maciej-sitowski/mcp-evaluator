# Integrations

## Slack

Connect Slack from Settings → Integrations → Slack, then choose a channel. Product Two posts a message when a check goes down and when it recovers.

## Webhooks

Webhooks send a JSON POST to your URL on every status change. Failed deliveries are retried 5 times over 30 minutes.

## API access

The REST API uses bearer tokens created under Settings → API. The rate limit is 100 requests per minute.
