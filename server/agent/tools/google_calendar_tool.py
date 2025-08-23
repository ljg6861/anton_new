from server.agent.tools.base_tool import BaseTool, ToolMetadata, ToolCapability
from typing import Dict, Any, Optional, List
import json
import httpx
import os
from datetime import datetime, timedelta
import asyncio
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class GoogleCalendarTool(BaseTool):
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    
    def __init__(self):
        super().__init__()
        self.service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Google Calendar API service with proper authentication"""
        creds = None
        
        # Load existing credentials
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        
        # If no valid credentials, trigger OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if os.path.exists('credentials.json'):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', self.SCOPES)
                    creds = flow.run_local_server(port=0)
                else:
                    raise Exception("Google Calendar credentials not found. Please provide credentials.json file.")
            
            # Save credentials for next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('calendar', 'v3', credentials=creds)

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name='google_calendar_tool',
            version='2.0.0',
            description='Complete Google Calendar API integration for creating, managing, and scheduling events',
            capabilities=[ToolCapability.EXTERNAL_API],
            author='Agent System',
            tags=['calendar', 'google', 'planning', 'scheduling']
        )

    def get_function_schema(self) -> Dict[str, Any]:
        return {
            'function': {
                'name': 'google_calendar_tool',
                'description': 'Complete Google Calendar API integration for creating, managing, and scheduling events',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'action': {
                            'type': 'string', 
                            'description': 'Action to perform',
                            'enum': ['create', 'get', 'update', 'delete', 'list', 'search', 'get_calendars']
                        },
                        'calendar_id': {
                            'type': 'string', 
                            'description': 'Calendar ID to operate on (default: primary)',
                            'default': 'primary'
                        },
                        'event_id': {
                            'type': 'string', 
                            'description': 'Event ID for get/update/delete operations'
                        },
                        'event_data': {
                            'type': 'object',
                            'description': 'Event data for create/update operations',
                            'properties': {
                                'summary': {'type': 'string', 'description': 'Event title'},
                                'description': {'type': 'string', 'description': 'Event description'},
                                'start': {'type': 'object', 'description': 'Start time (dateTime and timeZone)'},
                                'end': {'type': 'object', 'description': 'End time (dateTime and timeZone)'},
                                'attendees': {'type': 'array', 'description': 'List of attendee emails'},
                                'location': {'type': 'string', 'description': 'Event location'},
                                'reminders': {'type': 'object', 'description': 'Reminder settings'}
                            }
                        },
                        'query': {
                            'type': 'string',
                            'description': 'Search query for finding events'
                        },
                        'time_min': {
                            'type': 'string',
                            'description': 'Start time for listing events (RFC3339 format)'
                        },
                        'time_max': {
                            'type': 'string',
                            'description': 'End time for listing events (RFC3339 format)'
                        },
                        'max_results': {
                            'type': 'integer',
                            'description': 'Maximum number of results to return (default: 10)',
                            'default': 10
                        }
                    },
                    'required': ['action']
                }
            }
        }

    def run(self, arguments: Dict[str, Any]) -> str:
        try:
            if not self.service:
                self._initialize_service()
            
            action = arguments.get('action')
            calendar_id = arguments.get('calendar_id', 'primary')
            
            if action == 'create':
                return self._create_event(calendar_id, arguments.get('event_data', {}))
            elif action == 'get':
                return self._get_event(calendar_id, arguments.get('event_id'))
            elif action == 'update':
                return self._update_event(calendar_id, arguments.get('event_id'), arguments.get('event_data', {}))
            elif action == 'delete':
                return self._delete_event(calendar_id, arguments.get('event_id'))
            elif action == 'list':
                return self._list_events(
                    calendar_id, 
                    arguments.get('time_min'),
                    arguments.get('time_max'),
                    arguments.get('max_results', 10)
                )
            elif action == 'search':
                return self._search_events(calendar_id, arguments.get('query', ''))
            elif action == 'get_calendars':
                return self._get_calendars()
            else:
                return json.dumps({'status': 'error', 'message': f'Unknown action: {action}'})
                
        except Exception as e:
            return json.dumps({'status': 'error', 'message': f'Google Calendar API error: {str(e)}'})

    def _create_event(self, calendar_id: str, event_data: Dict[str, Any]) -> str:
        """Create a new calendar event"""
        try:
            # Build event object from provided data
            event = {}
            
            if 'summary' in event_data:
                event['summary'] = event_data['summary']
            if 'description' in event_data:
                event['description'] = event_data['description']
            if 'location' in event_data:
                event['location'] = event_data['location']
            
            # Handle start/end times
            if 'start' in event_data:
                event['start'] = event_data['start']
            if 'end' in event_data:
                event['end'] = event_data['end']
            
            # Handle attendees
            if 'attendees' in event_data:
                event['attendees'] = [{'email': email} for email in event_data['attendees']]
            
            # Handle reminders
            if 'reminders' in event_data:
                event['reminders'] = event_data['reminders']
            else:
                event['reminders'] = {'useDefault': True}
            
            created_event = self.service.events().insert(calendarId=calendar_id, body=event).execute()
            
            return json.dumps({
                'status': 'success',
                'event_id': created_event['id'],
                'html_link': created_event.get('htmlLink'),
                'summary': created_event.get('summary'),
                'start': created_event.get('start'),
                'end': created_event.get('end')
            })
            
        except HttpError as e:
            return json.dumps({'status': 'error', 'message': f'HTTP error: {e}'})

    def _get_event(self, calendar_id: str, event_id: str) -> str:
        """Retrieve a specific event"""
        if not event_id:
            return json.dumps({'status': 'error', 'message': 'Event ID is required for get operation'})
        
        try:
            event = self.service.events().get(calendarId=calendar_id, eventId=event_id).execute()
            
            return json.dumps({
                'status': 'success',
                'event': {
                    'id': event['id'],
                    'summary': event.get('summary', ''),
                    'description': event.get('description', ''),
                    'location': event.get('location', ''),
                    'start': event.get('start'),
                    'end': event.get('end'),
                    'attendees': event.get('attendees', []),
                    'html_link': event.get('htmlLink'),
                    'status': event.get('status')
                }
            })
            
        except HttpError as e:
            return json.dumps({'status': 'error', 'message': f'Event not found or HTTP error: {e}'})

    def _update_event(self, calendar_id: str, event_id: str, event_data: Dict[str, Any]) -> str:
        """Update an existing event"""
        if not event_id:
            return json.dumps({'status': 'error', 'message': 'Event ID is required for update operation'})
        
        try:
            # Get existing event
            event = self.service.events().get(calendarId=calendar_id, eventId=event_id).execute()
            
            # Update with new data
            if 'summary' in event_data:
                event['summary'] = event_data['summary']
            if 'description' in event_data:
                event['description'] = event_data['description']
            if 'location' in event_data:
                event['location'] = event_data['location']
            if 'start' in event_data:
                event['start'] = event_data['start']
            if 'end' in event_data:
                event['end'] = event_data['end']
            if 'attendees' in event_data:
                event['attendees'] = [{'email': email} for email in event_data['attendees']]
            
            updated_event = self.service.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()
            
            return json.dumps({
                'status': 'success',
                'event_id': updated_event['id'],
                'summary': updated_event.get('summary'),
                'updated': updated_event.get('updated')
            })
            
        except HttpError as e:
            return json.dumps({'status': 'error', 'message': f'Update failed or HTTP error: {e}'})

    def _delete_event(self, calendar_id: str, event_id: str) -> str:
        """Delete an event"""
        if not event_id:
            return json.dumps({'status': 'error', 'message': 'Event ID is required for delete operation'})
        
        try:
            self.service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            
            return json.dumps({
                'status': 'success',
                'message': f'Event {event_id} deleted from calendar {calendar_id}'
            })
            
        except HttpError as e:
            return json.dumps({'status': 'error', 'message': f'Delete failed or HTTP error: {e}'})

    def _list_events(self, calendar_id: str, time_min: Optional[str], time_max: Optional[str], max_results: int) -> str:
        """List events from calendar"""
        try:
            # Set default time range if not provided
            if not time_min:
                time_min = datetime.utcnow().isoformat() + 'Z'
            if not time_max:
                time_max = (datetime.utcnow() + timedelta(days=30)).isoformat() + 'Z'
            
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            event_list = []
            for event in events:
                event_list.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'start': event.get('start'),
                    'end': event.get('end'),
                    'location': event.get('location', ''),
                    'status': event.get('status')
                })
            
            return json.dumps({
                'status': 'success',
                'events': event_list,
                'total_count': len(event_list)
            })
            
        except HttpError as e:
            return json.dumps({'status': 'error', 'message': f'List failed or HTTP error: {e}'})

    def _search_events(self, calendar_id: str, query: str) -> str:
        """Search for events by query"""
        if not query:
            return json.dumps({'status': 'error', 'message': 'Query is required for search operation'})
        
        try:
            events_result = self.service.events().list(
                calendarId=calendar_id,
                q=query,
                maxResults=20,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            event_list = []
            for event in events:
                event_list.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'start': event.get('start'),
                    'end': event.get('end'),
                    'location': event.get('location', ''),
                    'description': event.get('description', '')
                })
            
            return json.dumps({
                'status': 'success',
                'query': query,
                'events': event_list,
                'total_count': len(event_list)
            })
            
        except HttpError as e:
            return json.dumps({'status': 'error', 'message': f'Search failed or HTTP error: {e}'})

    def _get_calendars(self) -> str:
        """Get list of available calendars"""
        try:
            calendar_list = self.service.calendarList().list().execute()
            calendars = calendar_list.get('items', [])
            
            calendar_info = []
            for calendar in calendars:
                calendar_info.append({
                    'id': calendar['id'],
                    'summary': calendar.get('summary', ''),
                    'description': calendar.get('description', ''),
                    'access_role': calendar.get('accessRole', ''),
                    'primary': calendar.get('primary', False)
                })
            
            return json.dumps({
                'status': 'success',
                'calendars': calendar_info,
                'total_count': len(calendar_info)
            })
            
        except HttpError as e:
            return json.dumps({'status': 'error', 'message': f'Calendar list failed or HTTP error: {e}'})
