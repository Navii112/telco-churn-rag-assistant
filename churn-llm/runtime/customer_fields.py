"""고객 관측 필드의 한국어 이름·단위·표시값."""
LABELS = {'tenure': ('이용 기간', '개월'), 'MonthlyCharges': ('월 요금', 'USD'),
          'TotalCharges': ('누적 요금', 'USD'), 'Contract': ('계약 유형', ''),
          'InternetService': ('인터넷 서비스', ''), 'PhoneService': ('전화 서비스', ''),
          'TechSupport': ('기술 지원 가입', ''), 'OnlineSecurity': ('온라인 보안 가입', ''),
          'PaymentMethod': ('결제 방법', '')}

TRANSLATIONS = {'Month-to-month': '월 단위 계약', 'Yes': '가입', 'No': '미가입',
                'Fiber optic': '광인터넷', 'Credit card (automatic)': '신용카드 자동 결제'}
