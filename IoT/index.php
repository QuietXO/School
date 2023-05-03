<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Support Page</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Poppins&display=swap"
		rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>Support Page</h1>
        <p>Feel free to contact us in case of any problems</p>
				<p>& we"ll reach out to you as soon as we can.</p>
        <form action="contact.php" method="POST">
						<div id="nameFirst">
	            <label for="nameF">Firt Name:</label>
	            <input type="text" name="nameF" id="nameF"
							placeholder="John" required>
						</div>
						<div id="nameLast">
							<label for="nameL">Last Name:</label>
	            <input type="text" name="nameL" id="nameL"
							placeholder="Doe">
						</div>

						<div id="mailE">
	            <label for="email">Email:</label>
	            <input type="email" name="email" id="email"
							placeholder="john.doe@email.com" required>
						</div>

						<label for="phone">Phone Number:</label>
						<div id="phoneCountry">
							<input type="text" name="country" id="country"
							placeholder="+421">
						</div>
						<div id="phoneNumber">
	            <input type="text" name="phone" id="phone"
							placeholder="9XX XXX XXX">
						</div>

						<div id="departmentRow">
							<div id="departmentText">
								<label for="department-selection">What seems to be the problem:</label>
							</div>
							<div id="departmentSelect">
								<select id="department-selection" name="concerned_department" required>
									<option value="">Select a Problem</option>
									<option value="logistics">Logistics</option>
									<option value="payment">Payment</option>
									<option value="technical">Technical Support</option>
								</select>
							</div>
						</div>

						<div id="messageSpace">
	            <label for="subject">Subject:</label>
	            <input type="text" name="subject" id="subject"
							placeholder="Subject" required>
	            <label for="message">Message</label>
	            <textarea name="message" cols="30" rows="10"
							placeholder="Message" required></textarea>
	            <input type="submit" value="Send">
						</div>
        </form>
    </div>
</body>
</html>
