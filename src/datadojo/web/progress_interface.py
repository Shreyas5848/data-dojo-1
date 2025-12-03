"""
Progress Tracking Dashboard
Track learning progress and achievements across sessions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import io


# Progress data file path
PROGRESS_FILE = Path("user_progress.json")


def load_progress_data():
    """Load progress data from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return get_default_progress()
    return get_default_progress()


def save_progress_data(data):
    """Save progress data to file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Could not save progress: {e}")


def get_default_progress():
    """Return default progress structure."""
    return {
        "user_info": {
            "name": "Data Scientist",
            "started": datetime.now().isoformat(),
            "level": 1,
            "xp": 0
        },
        "skills": {
            "data_exploration": {"level": 0, "xp": 0, "max_xp": 100},
            "data_cleaning": {"level": 0, "xp": 0, "max_xp": 100},
            "visualization": {"level": 0, "xp": 0, "max_xp": 100},
            "classification": {"level": 0, "xp": 0, "max_xp": 100},
            "regression": {"level": 0, "xp": 0, "max_xp": 100},
            "clustering": {"level": 0, "xp": 0, "max_xp": 100},
            "feature_engineering": {"level": 0, "xp": 0, "max_xp": 100},
            "time_series": {"level": 0, "xp": 0, "max_xp": 100}
        },
        "achievements": [],
        "activities": [],
        "notebooks_generated": 0,
        "datasets_profiled": 0,
        "datasets_explored": 0,
        "total_sessions": 0,
        "streak_days": 0,
        "last_active": None
    }


def calculate_level(xp):
    """Calculate level based on XP."""
    # Each level requires 100 * level XP
    level = 1
    xp_needed = 100
    total_xp = xp
    
    while total_xp >= xp_needed:
        total_xp -= xp_needed
        level += 1
        xp_needed = 100 * level
    
    return level, total_xp, xp_needed


def add_xp(progress, skill, amount, activity_name):
    """Add XP to a skill and record activity."""
    if skill in progress["skills"]:
        progress["skills"][skill]["xp"] += amount
        
        # Check for level up
        skill_data = progress["skills"][skill]
        while skill_data["xp"] >= skill_data["max_xp"]:
            skill_data["xp"] -= skill_data["max_xp"]
            skill_data["level"] += 1
            skill_data["max_xp"] = 100 * (skill_data["level"] + 1)
            
            # Add achievement for level up
            if skill_data["level"] in [1, 5, 10]:
                achievement = {
                    "name": f"{skill.replace('_', ' ').title()} Level {skill_data['level']}",
                    "description": f"Reached level {skill_data['level']} in {skill.replace('_', ' ')}",
                    "date": datetime.now().isoformat(),
                    "icon": "🏆"
                }
                progress["achievements"].append(achievement)
    
    # Add total XP
    progress["user_info"]["xp"] += amount
    level, _, _ = calculate_level(progress["user_info"]["xp"])
    progress["user_info"]["level"] = level
    
    # Record activity
    activity = {
        "name": activity_name,
        "skill": skill,
        "xp": amount,
        "date": datetime.now().isoformat()
    }
    progress["activities"].insert(0, activity)
    progress["activities"] = progress["activities"][:50]  # Keep last 50
    
    return progress


def check_achievements(progress):
    """Check and award achievements."""
    existing = [a["name"] for a in progress["achievements"]]
    
    # Notebooks achievements
    notebooks = progress["notebooks_generated"]
    if notebooks >= 1 and "First Notebook" not in existing:
        progress["achievements"].append({
            "name": "First Notebook",
            "description": "Generated your first analysis notebook",
            "date": datetime.now().isoformat(),
            "icon": "📓"
        })
    if notebooks >= 5 and "Notebook Ninja" not in existing:
        progress["achievements"].append({
            "name": "Notebook Ninja",
            "description": "Generated 5 analysis notebooks",
            "date": datetime.now().isoformat(),
            "icon": "🥷"
        })
    if notebooks >= 10 and "Notebook Master" not in existing:
        progress["achievements"].append({
            "name": "Notebook Master",
            "description": "Generated 10 analysis notebooks",
            "date": datetime.now().isoformat(),
            "icon": "🎓"
        })
    
    # Profiling achievements
    profiled = progress["datasets_profiled"]
    if profiled >= 1 and "Data Detective" not in existing:
        progress["achievements"].append({
            "name": "Data Detective",
            "description": "Profiled your first dataset",
            "date": datetime.now().isoformat(),
            "icon": "🔍"
        })
    if profiled >= 5 and "Data Analyst" not in existing:
        progress["achievements"].append({
            "name": "Data Analyst",
            "description": "Profiled 5 datasets",
            "date": datetime.now().isoformat(),
            "icon": "📊"
        })
    
    # Streak achievements
    streak = progress["streak_days"]
    if streak >= 3 and "3-Day Streak" not in existing:
        progress["achievements"].append({
            "name": "3-Day Streak",
            "description": "Practiced for 3 consecutive days",
            "date": datetime.now().isoformat(),
            "icon": "🔥"
        })
    if streak >= 7 and "Week Warrior" not in existing:
        progress["achievements"].append({
            "name": "Week Warrior",
            "description": "Practiced for 7 consecutive days",
            "date": datetime.now().isoformat(),
            "icon": "⚔️"
        })
    
    # Skill diversity
    skills_with_xp = sum(1 for s in progress["skills"].values() if s["xp"] > 0 or s["level"] > 0)
    if skills_with_xp >= 3 and "Well-Rounded" not in existing:
        progress["achievements"].append({
            "name": "Well-Rounded",
            "description": "Practiced 3 different skills",
            "date": datetime.now().isoformat(),
            "icon": "🌟"
        })
    if skills_with_xp >= 6 and "Renaissance Data Scientist" not in existing:
        progress["achievements"].append({
            "name": "Renaissance Data Scientist",
            "description": "Practiced 6 different skills",
            "date": datetime.now().isoformat(),
            "icon": "👑"
        })
    
    return progress


def update_streak(progress):
    """Update daily streak."""
    today = datetime.now().date()
    
    if progress["last_active"]:
        last_active = datetime.fromisoformat(progress["last_active"]).date()
        
        if last_active == today:
            # Already active today
            pass
        elif last_active == today - timedelta(days=1):
            # Consecutive day
            progress["streak_days"] += 1
        else:
            # Streak broken
            progress["streak_days"] = 1
    else:
        progress["streak_days"] = 1
    
    progress["last_active"] = datetime.now().isoformat()
    progress["total_sessions"] += 1
    
    return progress


def render_progress_dashboard():
    """Render the Progress Tracking Dashboard."""
    st.title("📊 Progress Tracking Dashboard")
    
    st.markdown("""
    Track your data science learning journey! Earn XP, level up skills, and unlock achievements.
    """)
    
    # Load progress
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = load_progress_data()
    
    progress = st.session_state.progress_data
    
    # Update streak on page load
    progress = update_streak(progress)
    progress = check_achievements(progress)
    save_progress_data(progress)
    
    # User Profile Section
    st.header("👤 Your Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    level, current_xp, next_level_xp = calculate_level(progress["user_info"]["xp"])
    
    with col1:
        st.metric("🎯 Level", level)
    with col2:
        st.metric("⭐ Total XP", progress["user_info"]["xp"])
    with col3:
        st.metric("🔥 Streak", f"{progress['streak_days']} days")
    with col4:
        st.metric("🏆 Achievements", len(progress["achievements"]))
    
    # XP Progress bar
    xp_progress = current_xp / next_level_xp
    st.progress(xp_progress)
    st.caption(f"Level {level} → Level {level + 1}: {current_xp}/{next_level_xp} XP")
    
    st.markdown("---")
    
    # Skills Overview
    st.header("🎓 Skills")
    
    skill_names = {
        "data_exploration": "📊 Data Exploration",
        "data_cleaning": "🧹 Data Cleaning",
        "visualization": "📈 Visualization",
        "classification": "🎯 Classification",
        "regression": "📉 Regression",
        "clustering": "🔮 Clustering",
        "feature_engineering": "🔧 Feature Engineering",
        "time_series": "📅 Time Series"
    }
    
    # Create skill cards in 2 columns
    cols = st.columns(2)
    
    for i, (skill_key, skill_name) in enumerate(skill_names.items()):
        with cols[i % 2]:
            skill_data = progress["skills"][skill_key]
            skill_level = skill_data["level"]
            skill_xp = skill_data["xp"]
            skill_max = skill_data["max_xp"]
            
            # Skill progress
            xp_pct = skill_xp / skill_max if skill_max > 0 else 0
            
            st.markdown(f"**{skill_name}** - Level {skill_level}")
            st.progress(xp_pct)
            st.caption(f"{skill_xp}/{skill_max} XP")
    
    st.markdown("---")
    
    # Skills Radar Chart
    st.header("📈 Skill Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create radar chart
        skill_levels = [progress["skills"][k]["level"] for k in skill_names.keys()]
        skill_labels = [v.split(" ", 1)[1] for v in skill_names.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=skill_levels + [skill_levels[0]],  # Close the shape
            theta=skill_labels + [skill_labels[0]],
            fill='toself',
            name='Your Skills',
            line_color='#FF9900',
            fillcolor='rgba(255, 153, 0, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(skill_levels) + 1] if skill_levels else [0, 5]
                )
            ),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Stats")
        st.metric("📓 Notebooks Generated", progress["notebooks_generated"])
        st.metric("🔍 Datasets Profiled", progress["datasets_profiled"])
        st.metric("📁 Datasets Explored", progress["datasets_explored"])
        st.metric("🎮 Total Sessions", progress["total_sessions"])
    
    st.markdown("---")
    
    # Achievements Section
    st.header("🏆 Achievements")
    
    if progress["achievements"]:
        # Show achievements in a grid
        achievements = progress["achievements"]
        
        cols = st.columns(4)
        for i, achievement in enumerate(achievements[:12]):  # Show max 12
            with cols[i % 4]:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #FF9900, #FFB366); border-radius: 10px; margin: 0.5rem 0;">
                    <span style="font-size: 2rem;">{achievement.get('icon', '🏆')}</span>
                    <p style="margin: 0.5rem 0 0 0; font-weight: bold; color: white; font-size: 0.9rem;">{achievement['name']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if len(achievements) > 12:
            with st.expander(f"View all {len(achievements)} achievements"):
                for achievement in achievements:
                    st.markdown(f"**{achievement.get('icon', '🏆')} {achievement['name']}** - {achievement['description']}")
    else:
        st.info("🎯 Complete activities to earn achievements!")
        
        # Show available achievements
        with st.expander("📋 Available Achievements"):
            st.markdown("""
            | Achievement | How to Unlock |
            |------------|---------------|
            | 📓 First Notebook | Generate your first analysis notebook |
            | 🔍 Data Detective | Profile your first dataset |
            | 🔥 3-Day Streak | Practice for 3 consecutive days |
            | 🌟 Well-Rounded | Practice 3 different skills |
            | 🥷 Notebook Ninja | Generate 5 analysis notebooks |
            | 📊 Data Analyst | Profile 5 datasets |
            | ⚔️ Week Warrior | Practice for 7 consecutive days |
            | 👑 Renaissance | Practice 6 different skills |
            """)
    
    st.markdown("---")
    
    # Recent Activity
    st.header("📜 Recent Activity")
    
    if progress["activities"]:
        for activity in progress["activities"][:10]:
            date_str = datetime.fromisoformat(activity["date"]).strftime("%b %d, %H:%M")
            st.markdown(f"• **{activity['name']}** (+{activity['xp']} XP) - {date_str}")
    else:
        st.info("No recent activity. Start exploring to earn XP!")
    
    st.markdown("---")
    
    # Quick Actions
    st.header("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Log Practice Session", type="primary"):
            # Add XP for general practice
            progress = add_xp(progress, "data_exploration", 10, "Practice Session")
            save_progress_data(progress)
            st.success("🎉 +10 XP for Data Exploration!")
            st.rerun()
    
    with col2:
        if st.button("📓 Log Notebook Creation"):
            progress["notebooks_generated"] += 1
            progress = add_xp(progress, "visualization", 25, "Created Notebook")
            progress = check_achievements(progress)
            save_progress_data(progress)
            st.success("🎉 +25 XP! Notebook logged!")
            st.rerun()
    
    with col3:
        if st.button("🔍 Log Dataset Profiled"):
            progress["datasets_profiled"] += 1
            progress = add_xp(progress, "data_exploration", 20, "Profiled Dataset")
            progress = check_achievements(progress)
            save_progress_data(progress)
            st.success("🎉 +20 XP! Dataset profiled!")
            st.rerun()
    
    st.markdown("---")
    
    # Log specific skill practice
    st.subheader("📝 Log Skill Practice")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        skill_to_log = st.selectbox(
            "Select Skill",
            options=list(skill_names.keys()),
            format_func=lambda x: skill_names[x]
        )
    
    with col2:
        activity_desc = st.text_input("Activity Description", placeholder="e.g., Practiced K-Means")
    
    with col3:
        st.write("")  # Spacing
        st.write("")
        if st.button("✅ Log Activity"):
            if activity_desc:
                progress = add_xp(progress, skill_to_log, 15, activity_desc)
                progress = check_achievements(progress)
                save_progress_data(progress)
                st.success(f"🎉 +15 XP for {skill_names[skill_to_log]}!")
                st.rerun()
            else:
                st.warning("Please enter an activity description")
    
    st.markdown("---")
    
    # Reset Progress (with confirmation)
    with st.expander("⚙️ Settings"):
        st.warning("⚠️ Danger Zone")
        
        if st.button("🗑️ Reset All Progress"):
            st.session_state.confirm_reset = True
        
        if st.session_state.get('confirm_reset', False):
            st.error("Are you sure? This will delete all your progress!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Reset Everything"):
                    st.session_state.progress_data = get_default_progress()
                    save_progress_data(st.session_state.progress_data)
                    st.session_state.confirm_reset = False
                    st.success("Progress reset!")
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()
    
    st.markdown("---")
    
    # Export Progress Report
    st.header("📤 Export Progress Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate text report
        report = generate_progress_report(progress, skill_names)
        
        st.download_button(
            label="📄 Download Text Report",
            data=report,
            file_name=f"datadojo_progress_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            help="Download a detailed text report of your progress"
        )
    
    with col2:
        # Generate JSON export
        json_export = json.dumps(progress, indent=2, default=str)
        
        st.download_button(
            label="📊 Download JSON Data",
            data=json_export,
            file_name=f"datadojo_progress_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            help="Download raw progress data in JSON format"
        )
    
    # Generate HTML report
    html_report = generate_html_report(progress, skill_names)
    
    st.download_button(
        label="🌐 Download HTML Report",
        data=html_report,
        file_name=f"datadojo_progress_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
        help="Download a printable HTML report (can be saved as PDF from browser)"
    )
    
    st.caption("💡 Tip: Open the HTML report in a browser and use Print → Save as PDF for a PDF version")


def generate_progress_report(progress, skill_names):
    """Generate a text progress report."""
    report = []
    report.append("=" * 60)
    report.append("📊 DATADOJO PROGRESS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    # User info
    report.append("👤 USER PROFILE")
    report.append("-" * 40)
    report.append(f"Level: {progress['user_info']['level']}")
    report.append(f"Total XP: {progress['user_info']['xp']}")
    report.append(f"Started: {progress['user_info'].get('started', 'Unknown')[:10]}")
    report.append(f"Current Streak: {progress.get('streak_days', 0)} days")
    report.append(f"Total Sessions: {progress.get('total_sessions', 0)}")
    report.append("")
    
    # Stats
    report.append("📈 STATISTICS")
    report.append("-" * 40)
    report.append(f"Notebooks Generated: {progress.get('notebooks_generated', 0)}")
    report.append(f"Datasets Profiled: {progress.get('datasets_profiled', 0)}")
    report.append(f"Datasets Explored: {progress.get('datasets_explored', 0)}")
    report.append("")
    
    # Skills
    report.append("🎓 SKILL LEVELS")
    report.append("-" * 40)
    for skill_key, skill_name in skill_names.items():
        skill = progress['skills'].get(skill_key, {"level": 0, "xp": 0})
        report.append(f"{skill_name}: Level {skill['level']} ({skill['xp']} XP)")
    report.append("")
    
    # Achievements
    report.append("🏆 ACHIEVEMENTS")
    report.append("-" * 40)
    achievements = progress.get('achievements', [])
    if achievements:
        for ach in achievements:
            report.append(f"• {ach.get('icon', '🏆')} {ach['name']}: {ach['description']}")
    else:
        report.append("No achievements yet. Keep practicing!")
    report.append("")
    
    # Recent activity
    report.append("📜 RECENT ACTIVITY (Last 10)")
    report.append("-" * 40)
    activities = progress.get('activities', [])[:10]
    if activities:
        for act in activities:
            date_str = act.get('date', '')[:10]
            report.append(f"• {act['name']} (+{act['xp']} XP) - {date_str}")
    else:
        report.append("No recent activity recorded.")
    report.append("")
    
    report.append("=" * 60)
    report.append("Keep learning and growing! 🚀")
    report.append("=" * 60)
    
    return "\n".join(report)


def generate_html_report(progress, skill_names):
    """Generate an HTML progress report that can be printed as PDF."""
    
    # Calculate skill levels for chart
    skill_levels = [(skill_names[k], progress['skills'].get(k, {}).get('level', 0)) for k in skill_names]
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataDojo Progress Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #FF9900, #FFB366);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #FF9900;
            border-bottom: 2px solid #FF9900;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #FF9900, #FFB366);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 2rem;
            font-weight: bold;
        }}
        .stat-card .label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        .skill-bar {{
            margin: 10px 0;
        }}
        .skill-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .skill-progress {{
            background: #eee;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .skill-fill {{
            background: linear-gradient(90deg, #FF9900, #FFB366);
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            font-size: 0.8rem;
        }}
        .achievement {{
            display: inline-block;
            background: linear-gradient(135deg, #FF9900, #FFB366);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 5px;
            font-size: 0.9rem;
        }}
        .activity {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .activity:last-child {{
            border-bottom: none;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
        }}
        @media print {{
            body {{ background: white; }}
            .section {{ box-shadow: none; border: 1px solid #ddd; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🥋 DataDojo Progress Report</h1>
        <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{progress['user_info']['level']}</div>
                <div class="label">Level</div>
            </div>
            <div class="stat-card">
                <div class="number">{progress['user_info']['xp']}</div>
                <div class="label">Total XP</div>
            </div>
            <div class="stat-card">
                <div class="number">{progress.get('streak_days', 0)}</div>
                <div class="label">Day Streak</div>
            </div>
            <div class="stat-card">
                <div class="number">{len(progress.get('achievements', []))}</div>
                <div class="label">Achievements</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Activity Stats</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{progress.get('notebooks_generated', 0)}</div>
                <div class="label">Notebooks</div>
            </div>
            <div class="stat-card">
                <div class="number">{progress.get('datasets_profiled', 0)}</div>
                <div class="label">Datasets Profiled</div>
            </div>
            <div class="stat-card">
                <div class="number">{progress.get('datasets_explored', 0)}</div>
                <div class="label">Datasets Explored</div>
            </div>
            <div class="stat-card">
                <div class="number">{progress.get('total_sessions', 0)}</div>
                <div class="label">Sessions</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🎓 Skills</h2>
        {''.join(f'''
        <div class="skill-bar">
            <div class="skill-name">{name}</div>
            <div class="skill-progress">
                <div class="skill-fill" style="width: {max(5, min(100, (level * 10 + 5)))}%">
                    Level {level}
                </div>
            </div>
        </div>
        ''' for name, level in skill_levels)}
    </div>
    
    <div class="section">
        <h2>🏆 Achievements</h2>
        {''.join(f"<span class='achievement'>{a.get('icon', '🏆')} {a['name']}</span>" for a in progress.get('achievements', [])) or '<p>No achievements yet. Keep practicing!</p>'}
    </div>
    
    <div class="section">
        <h2>📜 Recent Activity</h2>
        {''.join(f"<div class='activity'>• <strong>{a['name']}</strong> (+{a['xp']} XP) - {a.get('date', '')[:10]}</div>" for a in progress.get('activities', [])[:10]) or '<p>No recent activity recorded.</p>'}
    </div>
    
    <div class="footer">
        <p>🚀 Keep learning and growing with DataDojo!</p>
        <p><small>This report was generated automatically.</small></p>
    </div>
</body>
</html>
"""
    return html
