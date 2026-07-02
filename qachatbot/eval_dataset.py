# eval_dataset.py
#
# Hand-written evaluation set for the retriever. Each in-domain question is
# tied to the real handbook page it should retrieve (verified against the
# actual chunk titles/URLs produced by ingest.py). Out-of-domain questions
# are used to check that irrelevant queries get rejected before generation.

IN_DOMAIN = [
    ("How does the merge request buddy program work for new hires?",
     "https://handbook.gitlab.com/handbook/people-group/general-onboarding/mr-buddies/"),
    ("What are TaNewKi tips for new employees?",
     "https://handbook.gitlab.com/handbook/people-group/general-onboarding/tanewki-tips/"),
    ("How do GitLab Onboarding Buddies work?",
     "https://handbook.gitlab.com/handbook/people-group/general-onboarding/onboarding-buddies/"),
    ("What happens during offboarding at GitLab?",
     "https://handbook.gitlab.com/handbook/people-group/offboarding/"),
    ("What are frequently asked questions about offboarding?",
     "https://handbook.gitlab.com/handbook/people-group/offboarding/faq/"),
    ("What are GitLab's offboarding standards?",
     "https://handbook.gitlab.com/handbook/people-group/offboarding/offboarding_standards/"),
    ("What types of leave are available at GitLab?",
     "https://handbook.gitlab.com/handbook/people-group/time-off-and-absence/leave-types/"),
    ("What is GitLab's philosophy on taking time away from work?",
     "https://handbook.gitlab.com/handbook/people-group/time-off-and-absence/time-away-philosophy/"),
    ("What types of time off does GitLab offer?",
     "https://handbook.gitlab.com/handbook/people-group/time-off-and-absence/time-off-types/"),
    ("How does career development and mobility work at GitLab?",
     "https://handbook.gitlab.com/handbook/people-group/learning-and-development/career-development/"),
    ("What is an Individual Growth Plan (IGP)?",
     "https://handbook.gitlab.com/handbook/people-group/learning-and-development/career-development/igp-guide/"),
    ("What is the Growth and Development Fund?",
     "https://handbook.gitlab.com/handbook/people-group/learning-and-development/growth-and-development/"),
    ("What is the LevelUp program?",
     "https://handbook.gitlab.com/handbook/people-group/learning-and-development/level-up/"),
    ("How does mentoring work at GitLab?",
     "https://handbook.gitlab.com/handbook/people-group/learning-and-development/mentor/"),
    ("What benefits does GitLab offer through Modern Health?",
     "https://handbook.gitlab.com/handbook/total-rewards/benefits/modern-health/"),
    ("What is the Leave of Absence Toolkit for team members and managers?",
     "https://handbook.gitlab.com/handbook/total-rewards/benefits/parental-leave-toolkit/"),
    ("How does GitLab's annual compensation review cycle work?",
     "https://handbook.gitlab.com/handbook/total-rewards/compensation/compensation-review-cycle/"),
    ("What is GitLab's equity compensation program?",
     "https://handbook.gitlab.com/handbook/total-rewards/stock-options/"),
    ("What incentives does GitLab offer employees?",
     "https://handbook.gitlab.com/handbook/total-rewards/incentives/"),
    ("What is asynchronous communication and why does GitLab use it for remote work?",
     "https://handbook.gitlab.com/handbook/company/culture/all-remote/asynchronous/"),
    ("What is the definitive guide to remote internships about?",
     "https://handbook.gitlab.com/handbook/company/culture/all-remote/internship/"),
    ("How can remote workers combat burnout, isolation and anxiety?",
     "https://handbook.gitlab.com/handbook/company/culture/all-remote/mental-health/"),
]

OUT_OF_DOMAIN = [
    "What is the capital of France?",
    "How do I train a convolutional neural network in PyTorch?",
    "What's the weather like today?",
    "Explain how the TCP three-way handshake works.",
    "What is GitLab's current stock price?",
]
