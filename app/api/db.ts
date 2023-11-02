
import mysql from 'mysql2/promise'


export default async function excuteQuery({ query, values }: any) {
  try {
    const connection = await mysql.createConnection({
        host: process.env.DB_HOST,
        user: process.env.DB_USER,
        database: process.env.DB_NAME,
        password: process.env.DB_PASSWORD,
    });

    return connection.execute(query, values);
  } catch (error) {
    return { error };
  }
}