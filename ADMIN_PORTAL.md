# MedStats Admin Portal

## Overview

The admin portal provides real-time monitoring and management capabilities for the MedStats platform.

## Features

### 📊 System Overview Dashboard
- **User Statistics**: Total users, active users, professors, students
- **Analysis Metrics**: Total, running, completed, and failed analyses
- **Activity Tracking**: Analyses submitted today and this week
- **Performance Metrics**: Average analysis completion time

### 🚨 Problem Detection
- **Stuck Analyses Alert**: Automatically detects analyses running longer than 30 minutes
- **Failed Analyses Table**: Shows recent failures from the last 24 hours
- **Real-time Monitoring**: Auto-refreshes every 60 seconds

### 📋 Activity Feed
- Last 50 analysis activities across all users
- User details (name, email, role)
- Analysis status and timestamps
- Quick links to detailed job views

### 👥 User Management
- View all users (professors and students)
- Filter by role and status
- Create new users
- Activate/deactivate accounts

### 🔧 System Health
- Database connection status
- Redis connectivity
- RAG corpus statistics
- Queue depth monitoring

## Access

### Default Admin Credentials

**URL**: `http://localhost:3000/admin` (or your deployment URL + `/admin`)

**Login**:
- Email: `admin@medstats.io`
- Password: `MedstatsAdmin@10`

> ⚠️ **IMPORTANT**: Change this password in production by updating the `seed_admin.py` script before deployment.

### Creating the Admin User

The admin user is automatically created when you deploy the application. The seeding happens in the Docker entrypoint script.

**Manual seeding** (if needed):
```bash
cd /path/to/medstats-api
python scripts/seed_admin.py
```

## Navigation

Once logged in, the admin portal has 5 main sections:

1. **Overview** - Dashboard with aggregate statistics and recent activity
2. **Jobs** - Background job monitoring and management
3. **Submissions** - All user submissions across the platform
4. **Users** - User management (create, edit, deactivate)
5. **System** - System health and RAG corpus management

## API Endpoints

The admin portal is powered by these API endpoints:

- `GET /admin/overview/stats` - Aggregate statistics
- `GET /admin/overview/activity` - Recent analysis activity (last 50)
- `GET /admin/overview/stuck` - Stuck analyses (>30 min)
- `GET /admin/overview/failed` - Failed analyses (last 24h)
- `GET /admin/jobs` - List background jobs
- `GET /admin/submissions` - List all submissions
- `GET /admin/users` - List all users
- `GET /admin/system/health` - System health check
- `GET /admin/rag/stats` - RAG corpus statistics

## Security

- Only users with `role: ADMIN` can access the admin portal
- All admin routes require authentication
- 403 Forbidden returned for non-admin users
- Admin role is defined in `UserRole` enum

## Development

To test the admin portal locally:

1. Start the backend:
   ```bash
   cd medstats-api
   ./scripts/deploy.sh
   ```

2. Start the frontend:
   ```bash
   cd medstats-ui
   ./start-dev.sh
   ```

3. Navigate to `http://localhost:3000/admin`

4. Login with admin credentials

## Customization

### Changing Admin Password

Edit `/scripts/seed_admin.py`:
```python
admin_password = "YourNewPassword123!"
```

Then run:
```bash
python scripts/seed_admin.py
```

### Adding Admin Users

You can create additional admin users via:

1. **UI**: Login as admin → Users → Create User → Set role to "Admin"
2. **Script**: Modify `seed_admin.py` to create multiple admin users
3. **Database**: Directly insert into `users` table with `role = 'admin'`

## Monitoring Best Practices

1. **Check Overview Daily**: Review stuck and failed analyses
2. **Monitor Activity Feed**: Identify usage patterns and bottlenecks
3. **Watch System Health**: Ensure RAG corpus is indexed and Redis is responsive
4. **Review Failed Jobs**: Investigate and fix recurring failures
5. **Track Performance**: Monitor average completion time trends

## Troubleshooting

### Can't Access Admin Portal
- Verify user has `role: ADMIN` in database
- Check JWT token is valid
- Clear browser localStorage and re-login

### Stuck Analyses Not Clearing
- Check worker logs: `docker compose logs -f worker`
- Verify Celery is processing jobs
- Review Redis queue depth in System tab

### Data Not Refreshing
- Auto-refresh runs every 60 seconds
- Click "Refresh" button for immediate update
- Check browser console for API errors

## Reference

The admin portal is inspired by the Shopify Translator admin interface and follows similar patterns for monitoring, user management, and system health tracking.
