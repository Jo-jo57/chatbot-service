import random
from datetime import datetime

class StudentChatbot:
    def __init__(self):
        self.knowledge_base = {
            'campus_locations': {
                'academic_buildings': [
                    "The main academic buildings are typically located in the central campus area. You can find the Engineering Building near the Science Complex, and the Business School is usually close to the Administration Building.",
                    "Most lecture halls and classrooms are numbered by building and floor. For example, 'ENG 201' means Engineering Building, 2nd floor, room 01.",
                    "The Library is usually centrally located and serves as a good landmark for navigation. Most academic departments have offices in or near their respective buildings."
                ],
                'dining_facilities': [
                    "The main cafeteria is typically open from 7 AM to 9 PM on weekdays. There are usually several dining options including fast food, healthy choices, and vegetarian meals.",
                    "Most universities have multiple dining locations: main cafeteria, student union food court, coffee shops, and sometimes food trucks on campus.",
                    "You can usually pay with your student ID card, cash, or card at most dining facilities. Many schools offer meal plans for convenience."
                ],
                'dormitories': [
                    "Residence halls are typically organized by year or theme. Freshman dorms are often closer to dining and academic buildings.",
                    "Most dorm buildings have common areas, laundry facilities, and study rooms. RAs (Resident Advisors) live on each floor to help students.",
                    "Visiting hours and guest policies vary by dorm. Check with your RA or housing office for specific rules about overnight guests."
                ]
            },
            'academic_support': {
                'registration': [
                    "Course registration usually happens online through the student portal. Priority registration is often given to seniors, then juniors, etc.",
                    "Make sure to check prerequisites before registering. Some courses require permission from the instructor or department.",
                    "If a class is full, you can often join a waitlist. Attend the first class anyway - professors sometimes add students who show up."
                ],
                'library_services': [
                    "The library offers study spaces, computer labs, printing services, and research assistance. Librarians can help you find academic sources for papers.",
                    "Most libraries have quiet study areas, group study rooms you can reserve, and 24/7 access during finals week.",
                    "You can usually borrow books, access online databases, and get help with citations and research methods."
                ],
                'tutoring': [
                    "Many universities offer free tutoring through the Academic Success Center or similar departments.",
                    "Peer tutoring programs connect you with other students who've excelled in specific subjects.",
                    "Some departments have dedicated tutoring for challenging courses like calculus, chemistry, or writing."
                ]
            },
            'campus_services': {
                'health_center': [
                    "The campus health center typically provides basic medical care, mental health counseling, and wellness programs.",
                    "Most health services are covered by student fees, but some specialized services may have additional costs.",
                    "You can usually make appointments online or by phone. Emergency services are available 24/7."
                ],
                'transportation': [
                    "Many campuses have shuttle services that run between different campus locations and nearby off-campus areas.",
                    "Parking permits are usually required and can be purchased through the parking services office. Different lots have different permit requirements.",
                    "Some universities offer bike sharing programs or have bike repair stations around campus."
                ],
                'student_activities': [
                    "The Student Activities Center coordinates clubs, events, and leadership opportunities. They usually have a database of all student organizations.",
                    "Intramural sports, cultural events, and social activities are great ways to meet people and get involved on campus.",
                    "Many schools host orientation events, career fairs, and special programs throughout the year."
                ]
            },
            'practical_tips': {
                'campus_life': [
                    "Get a campus map app or physical map during your first week. Most schools have digital maps with real-time information.",
                    "Join study groups and attend office hours - professors and TAs are there to help you succeed.",
                    "Take advantage of free campus resources like the gym, events, and student support services."
                ]
            }
        }

    def generate_response(self, question, topic=None, urgency=None):
        """
        Generate a response based on the user's question and any provided context.
        """
        if topic is None:
            topic = self.extract_topic(question)

        if topic in self.knowledge_base:
            return self.knowledge_base[topic]
        else:
            return "I'm sorry, I don't have any information about that topic. Please try asking about something related to campus services, like the library, registration, or dining options."

    def extract_topic(self, question):
        """
        Extract the topic from the user's question.
        """
        # Implement logic to determine the topic based on the question
        # This can be done using natural language processing techniques, keyword matching, etc.
        
        # For now, we'll just use a simple approach:
        words = question.lower().split()
        for word in words:
            if word in self.knowledge_base:
                return word
        
        return None

    def get_helpful_links(self, question):
        """
        Provide a list of helpful links based on the user's question.
        """
        # Implement logic to determine relevant links based on the question
        # This can be done by matching keywords, understanding the intent, etc.
        
        # For now, we'll just return some sample links:
        return [
            "https://www.campusmap.edu",
            "https://www.campusservices.edu",
            "https://www.campusregistration.edu"
        ]

    def get_current_time(self):
        """
        Get the current time for the chatbot's responses.
        """
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

